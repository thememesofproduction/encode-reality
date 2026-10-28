# What a reader should take from each figure

This is the acceptance test for the post's figures and their explanations. A reader with no maths beyond
school, who reads the post and uses the figures, should come away with the takeaway and be able to answer
the questions. The naive-reader check gives a fresh reader the rendered post and these questions, without
the answers below. Any question they get wrong marks a gap in the post, not in the reader.

## 1. Instrument: "The answer is off the line"

**Takeaway.** Searching harder along the real number line can't solve the equation, because no real number
solves it. The solutions are off the line, in a direction the real numbers don't have.

1. What does a lower number in the readout mean, and what number would mean "solved"?
   *Answer:* |p| measures how far p(x) is from zero; 0 means solved.
2. Confined to the line, where do the searchers end up, and is that the best they could do?
   *Answer:* They pile up at about 9.77, the lowest the line allows. Some get stuck at about 32.5. Searching
   longer doesn't help, because 9.77 is the minimum on the line.
3. What changes when the second direction is allowed, and why?
   *Answer:* The searchers reach 0, at the four roots. The roots were never on the line; they need the
   imaginary direction.

## 2. Figure: searching harder vs changing the space (Plotly)

**Takeaway.** However long a search confined to the real numbers runs, it never goes below the floor. The
Level-2 system, which searches over number systems, drops far below it once it finds one with a square root
of −1.

1. What does the horizontal axis count, and what does "log scale" mean here?
   *Answer:* Guesses checked. Each step along the axis is ten times the last.
2. What does the height 1 on the vertical axis mean?
   *Answer:* The best possible on the real line (the floor). Lower is better.
3. Which line goes below 1, and roughly when?
   *Answer:* Level 2's line, the one that changes the space, after about ten thousand guesses (median about
   12,000).

## 3. Instrument: "Choose a number system"

**Takeaway.** Each point is a different number system. Only one region, the one under the curve, contains a
square root of −1, and only there does a search get below the floor. Every system in that region is really
the complex numbers in disguise.

1. What does moving the point change?
   *Answer:* The rule e² = α + βe, so which number system the search runs in.
2. Where does a search break the floor, and what do those systems have in common?
   *Answer:* Below the dashed curve (Δ < 0), where they contain a square root of −1. They are all the complex
   numbers in disguise.
3. In the replayed run, does Level 2 creep toward that region step by step, or land in it?
   *Answer:* It lands in it by proposing distant systems and recognising success. Nothing outside the region
   is closer to the answer than anything else.

## 4. Figure: how often each strategy finds the highest point (Plotly)

**Takeaway.** On a hard landscape, how you search matters a lot, but the answer was always in the space.
Strategies differ in whether and how fast they reach the highest point, never in where it is.

1. What does a line reaching 100% mean?
   *Answer:* Every run of that strategy got within 1% of the highest point by that many guesses.
2. Which strategy was the most reliable, and was the "learning which strategy to use" approach better than
   switching at random?
   *Answer:* Fixed restarts (99–100%). Learning which strategy to use was not measurably better than random
   switching.
3. How is this failure different from the one in figure 2?
   *Answer:* Here the failure is in the search, and more exploration fixes it. In figure 2 the failure is in
   the space, and only a change of space fixes it.

## 5. The argument about AI

**Takeaway.** The line between Level 1 and Level 2 is what the system acts on, candidates or generators. It
is not whether the system loops back on itself. Recursion lets a system keep climbing levels without a
designer, but even self-improving systems keep a judge they do not rewrite.

1. Is a language model inside a self-improving system Level 1 or Level 2, and why?
   *Answer:* The system can be Level 2 when it changes its own prompts, tools, code or weights, because it is
   then acting on generators. The verdict belongs to the whole system, not the model alone.
2. Name one example from the post of something that loops back on itself or adjusts itself and still stays
   at Level 1.
   *Answer:* Any one of these: a hill-climber building each guess from its last; a language model feeding
   its output back in; a search on the real line that rewrote its own step rule, which would still guess
   only real numbers; or the portfolio that chooses among strategies and still stops at the floor.
3. What do current self-improving systems almost never change, and why does that matter?
   *Answer:* Their judge (the evaluator or benchmark), and usually the base model. A system that rewrote its
   own judge could declare success. In the Darwin Gödel Machine, keeping variants that don't score best yet was worth more
   (27 points) than self-modification (11 points).
