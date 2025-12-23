// /app/api/chat/route.ts
import {
  OpenAIStream,
  StreamingTextResponse,
  experimental_StreamData,
} from 'ai'; // <--- Vercel AI SDK v3 imports
import OpenAI from 'openai';
import { NextRequest, NextResponse } from 'next/server';

// Import our refactored orchestrator functions
import {
  orchestratePreamble,
  generateFollowups,
  COACH_SYSTEM,
  Message,
} from '@/lib/orchestrator';

// Import findFeaturedMatching from its correct source file
import { findFeaturedMatching } from '@/lib/marketplace';

export const runtime = 'nodejs'; // Must be nodejs for supabaseAdmin
export const dynamic = 'force-dynamic';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Helper to provide default followups on error
function defaultFollowups(): string[] {
  return [
    'Find local apprenticeships',
    'Explore training programs',
    'Compare typical salaries (BLS)',
  ].slice(0, 3);
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { messages, location } = body;
  
  let messagesForLLM: Message[] = []; // Define here to have scope
  let lastUserRaw = '';
  let effectiveLocation: string | null = null;
  let internalRAG = '';

  try {
    // 1. Run all "pre-work" (RAG, context building, checks)
    const preambleResult = await orchestratePreamble({ messages, location });
    
    // Destructure results
    messagesForLLM = preambleResult.messagesForLLM;
    lastUserRaw = preambleResult.lastUserRaw;
    effectiveLocation = preambleResult.effectiveLocation;
    internalRAG = preambleResult.internalRAG;
    
    // 2. Handle guard conditions
    if (preambleResult.domainGuarded) {
      return NextResponse.json({
        answer:
          'I focus on modern manufacturing careers. We can explore roles like CNC Machinist, Robotics Technician, Welding Programmer, Additive Manufacturing, Maintenance Tech, or Quality Control.',
        followups: defaultFollowups(),
      });
    }

    // 3. Prepare the final LLM call
    const systemMessages: Message[] = [
      { id: 'system_prompt', role: 'system', content: COACH_SYSTEM },
    ];
    if (effectiveLocation) {
      systemMessages.push({
        id: 'system_location',
        role: 'system',
        content: `User location: ${effectiveLocation}`,
      });
    }

    // 4. Create the OpenAI stream
    const responseStream = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      temperature: 0.3,
      messages: [...systemMessages, ...messagesForLLM] as any, // Fix 1
      stream: true,
    });

    // 5. Initialize Vercel AI SDK StreamData
    const data = new experimental_StreamData();

    // 6. Convert the OpenAI stream to the Vercel AI SDK stream (v3 style)
    const stream = OpenAIStream(responseStream as any, { // Fix 2
      onFinal: async (completion) => {
        // This logic runs *after* the stream is done
        
        // a. Find Featured Listings
        let answerWithFeatured = completion;
        try {
          const featured = await findFeaturedMatching(
            lastUserRaw,
            effectiveLocation ?? undefined // Fix 3
          );
          if (Array.isArray(featured) && featured.length > 0) {
            const locTxt = effectiveLocation ? ` near ${effectiveLocation}` : '';
            const lines = featured
              .map((f: any) => `- **${f.title}** — ${f.org} (${f.location})`)
              .join('\n');
            answerWithFeatured += `\n\n**Featured${locTxt}:**\n${lines}`;
          }
        } catch (err) {
          console.error('Error finding featured items:', err);
        }

        // b. Add "Next Steps"
        let finalAnswerWithSteps = answerWithFeatured;
        if (internalRAG) {
          // Check if we triggered the internal search
          finalAnswerWithSteps = finalAnswerWithSteps.replace(
            /(\n\n\*\*Next Steps:\*\*.*)/is,
            ''
          );
          finalAnswerWithSteps += `\n\n**Next Steps**
You can also search for more opportunities on your own:
* [Search SkillStrong Programs](/programs/all)
* [Search SkillStrong Jobs](/jobs/all)
* [Search US Department of Education for programs](https://collegescorecard.ed.gov/)
* [Search for jobs on Indeed.com](https://www.indeed.com/)`;
        }

        // c. Generate Followups (FIX #2)
        // We pass the full message history that the LLM used
        const finalMessages: Message[] = [
            ...messagesForLLM, 
            { id: 'final_answer', role: 'assistant', content: finalAnswerWithSteps }
        ];
        
        const followups = await generateFollowups(
          finalMessages, // Pass full context
          finalAnswerWithSteps,
          effectiveLocation ?? undefined
        );

        // d. Append final data to the StreamData
        data.append(JSON.stringify({ // v3 expects a string
          finalAnswer: finalAnswerWithSteps,
          followups: followups,
        }));

        // e. Close the StreamData
        data.close();

      },
      experimental_streamData: true, // Tell it to use the data stream
    });

    // 7. Return the streaming response
    return new StreamingTextResponse(stream, {}, data);
    
  } catch (e: any) {
    if (e.message === 'LOCATION_REQUIRED') {
      // Handle the specific "location missing" error
      return NextResponse.json({
        answer:
          'To find local results, please set your location using the button in the header.',
        followups: [], // Send empty followups
      });
    }

    console.error("Error in /api/chat route:", e);
    // Generate followups even on error
     const errorFollowups = await generateFollowups(
          messagesForLLM, // Pass whatever context we had
          "Sorry, I couldn't process that.",
          effectiveLocation ?? undefined
     );
    return NextResponse.json(
      { 
          answer: "Sorry, I couldn't process that.", 
          followups: errorFollowups.length > 0 ? errorFollowups : defaultFollowups()
      },
      { status: 500 }
    );
  }
}
