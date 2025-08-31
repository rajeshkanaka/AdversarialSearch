
import React from 'react';
import Card from '../ui/Card';
import CodeBlock from '../ui/CodeBlock';

const hMinimaxCode = `
function H-MINIMAX(state, depth) {
  if (CUTOFF-TEST(state, depth)) {
    return EVAL(state);
  }

  if (PLAYER(state) == MAX) {
    // ... same as minimax ...
  } else {
    // ... same as minimax ...
  }
}
`;

const ImperfectDecisionsSection: React.FC = () => {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Imperfect Real-Time Decisions</h1>
        <p className="text-xl text-slate-600">Making good moves when you can't search the entire game tree.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">The Problem with Full-Width Search</h2>
        <p className="text-slate-700 leading-relaxed">
          For complex games like chess, the game tree is enormous (over 10⁴⁰ nodes). Minimax and even Alpha-Beta pruning cannot search to the end of the game in a reasonable amount of time.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          To solve this, programs must make <strong className="font-semibold">imperfect decisions</strong>. We do this by cutting the search off early and applying a heuristic <strong className="font-semibold">evaluation function</strong> to the "leaf" nodes of our partial search tree.
        </p>
        <ul className="list-disc list-inside mt-4 space-y-2 text-slate-700">
            <li><strong>Cutoff Test:</strong> A function that decides when to stop searching deeper. This is often a simple depth limit.</li>
            <li><strong>Evaluation Function (EVAL):</strong> An algorithm that provides an estimate of the utility of a non-terminal game state. For example, in chess, this could be based on material advantage (pawns=1, queen=9, etc.).</li>
        </ul>
      </Card>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Heuristic Minimax</h2>
        <p className="text-slate-700 mb-4">
          The Minimax algorithm is modified to use the cutoff test and evaluation function. Instead of checking for a terminal state, it checks the cutoff condition. Instead of using a utility function at the end, it uses the heuristic evaluation function.
        </p>
        <CodeBlock code={hMinimaxCode} />
      </Card>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Common Pitfalls</h2>
        <p className="text-slate-700 leading-relaxed">
          Using a fixed-depth cutoff can lead to serious errors.
        </p>
        <h3 className="text-lg font-semibold mt-4 mb-2">Quiescence and Volatility</h3>
        <p className="text-slate-700 leading-relaxed">
          An evaluation function should only be applied to <strong className="font-semibold">quiescent</strong> positions—those that are not in the middle of a volatile exchange (like a series of captures). If a search cuts off mid-capture, the evaluation will be completely wrong. A common solution is <strong className="font-semibold">quiescence search</strong>, which extends the search beyond the depth limit for volatile moves like captures until a stable position is reached.
        </p>
        <h3 className="text-lg font-semibold mt-4 mb-2">The Horizon Effect</h3>
        <p className="text-slate-700 leading-relaxed">
          This occurs when a program faces an unavoidable negative event. It may make delaying moves to push the event "over the horizon" of its search depth, even if these moves are ultimately futile. For example, sacrificing pawns to delay the capture of a queen for a few moves. The search sees the pawn sacrifice as good because it doesn't see the eventual queen capture within its depth limit.
        </p>
        <div className="mt-4 p-4 border rounded-lg bg-slate-50 flex justify-center">
            {/* Simple diagram illustrating horizon effect */}
            <p className="text-center text-slate-600">Imagine a search depth of 4 moves.<br/>
            <strong className="text-red-500">Path A:</strong> Lose Queen in 5 moves.<br/>
            <strong className="text-green-500">Path B:</strong> Lose a Pawn now, but lose Queen in 6 moves.<br/>
            The algorithm incorrectly chooses Path B because the loss of the Queen is beyond its search "horizon".
            </p>
        </div>
      </Card>
    </div>
  );
};

export default ImperfectDecisionsSection;
