
import React from 'react';
import Card from '../ui/Card';
import CodeBlock from '../ui/CodeBlock';

const expectiminimaxCode = `
function EXPECTIMINIMAX(state) {
  if (TERMINAL-TEST(state)) {
    return UTILITY(state);
  }
  
  const player = PLAYER(state);

  if (player == MAX) {
    return max(ACTIONS(state).map(a => EXPECTIMINIMAX(RESULT(state, a))));
  }
  
  if (player == MIN) {
    return min(ACTIONS(state).map(a => EXPECTIMINIMAX(RESULT(state, a))));
  }

  if (player == CHANCE) {
    let sum = 0;
    const possibleOutcomes = GET-OUTCOMES(state);
    for (let outcome of possibleOutcomes) {
      sum += P(outcome) * EXPECTIMINIMAX(RESULT(state, outcome));
    }
    return sum;
  }
}
`;

const StochasticGamesSection: React.FC = () => {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Stochastic Games</h1>
        <p className="text-xl text-slate-600">Making decisions in games with an element of chance, like dice rolls.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Introducing Uncertainty</h2>
        <p className="text-slate-700 leading-relaxed">
          Many real-world games, such as Backgammon or Monopoly, involve a random element. These are called <strong className="font-semibold">stochastic games</strong>. The unpredictability of events like dice rolls means we can no longer construct a simple game tree of MAX and MIN nodes.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          To handle this, we introduce a third type of node: the <strong className="font-semibold text-purple-600">CHANCE node</strong>. These nodes represent random events, and their branches correspond to the possible outcomes of that event, each with a specific probability.
        </p>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">The Expectiminimax Algorithm</h2>
        <p className="text-slate-700 leading-relaxed">
          For games with chance nodes, we can't calculate a definite minimax value. Instead, we calculate the <strong className="font-semibold">expected value</strong> of a position—the average utility over all possible outcomes of the chance events.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          The Expectiminimax algorithm generalizes Minimax:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-slate-700">
          <li><strong className="font-semibold text-green-600">MAX nodes</strong> compute the maximum of their successors' values.</li>
          <li><strong className="font-semibold text-red-600">MIN nodes</strong> compute the minimum of their successors' values.</li>
          <li><strong className="font-semibold text-purple-600">CHANCE nodes</strong> compute the weighted average of their successors' values, based on the probability of each outcome.</li>
        </ul>
        <div className="flex justify-center p-4 mt-4 border rounded-lg bg-slate-50">
            <svg width="400" height="250" viewBox="0 0 400 250" className="font-sans text-sm">
                <title>Expectiminimax Tree (based on Fig 5.11)</title>
                {/* MAX Node */}
                <polygon points="200,10 215,40 185,40" fill="#a7f3d0" stroke="#10b981" />
                <text x="200" y="30" textAnchor="middle">MAX</text>

                {/* Chance Node */}
                <line x1="200" y1="40" x2="200" y2="60" stroke="#9ca3af"/>
                <circle cx="200" cy="75" r="15" fill="#e9d5ff" stroke="#8b5cf6" />
                <text x="200" y="79" textAnchor="middle">CHANCE</text>

                {/* MIN Nodes */}
                <line x1="200" y1="90" x2="100" y2="120" stroke="#9ca3af" />
                <text x="145" y="105" fontSize="10">Roll 1-1 (p=1/36)</text>
                <polygon points="100,120 115,150 85,150" fill="#fecaca" stroke="#ef4444" transform="rotate(180 100 135)"/>
                <text x="100" y="139" textAnchor="middle">MIN</text>

                <line x1="200" y1="90" x2="300" y2="120" stroke="#9ca3af"/>
                <text x="255" y="105" fontSize="10">Roll 6-5 (p=1/18)</text>
                <polygon points="300,120 315,150 285,150" fill="#fecaca" stroke="#ef4444" transform="rotate(180 300 135)"/>
                <text x="300" y="139" textAnchor="middle">MIN</text>

                {/* Terminal Nodes */}
                <line x1="100" y1="150" x2="70" y2="180" stroke="#9ca3af"/>
                <line x1="100" y1="150" x2="130" y2="180" stroke="#9ca3af"/>
                <rect x="60" y="190" width="20" height="20" fill="#e5e7eb" />
                <text x="70" y="204" textAnchor="middle">2</text>
                <rect x="120" y="190" width="20" height="20" fill="#e5e7eb" />
                <text x="130" y="204" textAnchor="middle">-1</text>

                 <line x1="300" y1="150" x2="270" y2="180" stroke="#9ca3af"/>
                <line x1="300" y1="150" x2="330" y2="180" stroke="#9ca3af"/>
                <rect x="260" y="190" width="20" height="20" fill="#e5e7eb" />
                <text x="270" y="204" textAnchor="middle">1</text>
                <rect x="320" y="190" width="20" height="20" fill="#e5e7eb" />
                <text x="330" y="204" textAnchor="middle">-1</text>
            </svg>
        </div>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">Pseudocode</h2>
        <CodeBlock code={expectiminimaxCode} />
      </Card>
    </div>
  );
};

export default StochasticGamesSection;
