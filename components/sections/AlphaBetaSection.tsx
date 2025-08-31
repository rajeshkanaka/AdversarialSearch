
import React from 'react';
import Card from '../ui/Card';
import Button from '../ui/Button';
import CodeBlock from '../ui/CodeBlock';
import GameTreeVisualizer from '../interactive/GameTreeVisualizer';
import { useAlphaBeta } from '../../hooks/useAlphaBeta';
import { GAME_TREE_DATA } from '../../constants';

const alphaBetaCode = `
function alphaBeta(node, depth, α, β, maximizingPlayer) {
  if (depth === 0 || node.isTerminal) {
    return node.utility;
  }

  if (maximizingPlayer) {
    let value = -Infinity;
    for (let child of node.children) {
      value = Math.max(value, alphaBeta(child, depth - 1, α, β, false));
      α = Math.max(α, value);
      if (α >= β) {
        break; // β cutoff
      }
    }
    return value;
  } else { // Minimizing player
    let value = +Infinity;
    for (let child of node.children) {
      value = Math.min(value, alphaBeta(child, depth - 1, α, β, true));
      β = Math.min(β, value);
      if (β <= α) {
        break; // α cutoff
      }
    }
    return value;
  }
}
`;

const AlphaBetaSection: React.FC = () => {
  const { tree, steps, currentStep, run, reset, isRunning } = useAlphaBeta(GAME_TREE_DATA);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Alpha-Beta Pruning</h1>
        <p className="text-xl text-slate-600">An optimization for Minimax that avoids searching irrelevant branches.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Core Concept</h2>
        <p className="text-slate-700 leading-relaxed">
          Alpha-Beta pruning is a search algorithm that seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree. It stops evaluating a move when it has found proof that the move is worse than a previously examined move. Such moves need not be evaluated further.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          It maintains two values, alpha and beta, which represent the best outcomes found so far for MAX and MIN, respectively, along the path to the root:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-slate-700">
          <li><strong className="font-semibold text-green-600">Alpha (α):</strong> The best value (highest) that MAX can currently guarantee at that level or above. Initialized to -∞.</li>
          <li><strong className="font-semibold text-red-600">Beta (β):</strong> The best value (lowest) that MIN can currently guarantee at that level or above. Initialized to +∞.</li>
        </ul>
        <p className="mt-4 text-slate-700 leading-relaxed">
          Pruning occurs when <strong className="font-semibold">alpha becomes greater than or equal to beta (α ≥ β)</strong>. This means that a player has found a better path earlier in the search, so the current branch can be ignored.
        </p>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-3">Interactive Visualization</h2>
        <p className="text-slate-600 mb-4">
          Run the visualization to see how Alpha-Beta pruning explores the same tree but "prunes" (greys out) branches it doesn't need to visit. Compare the number of nodes visited to the standard Minimax algorithm.
        </p>
        <div className="flex space-x-4 mb-4">
          <Button onClick={run} disabled={isRunning}>
            {isRunning ? 'Running...' : 'Run Visualization'}
          </Button>
          <Button onClick={reset} variant="secondary" disabled={isRunning || currentStep === 0}>
            Reset
          </Button>
        </div>
        
        <div className="p-4 border rounded-lg bg-slate-50 min-h-[320px]">
          <GameTreeVisualizer tree={tree} showAlphaBeta={true} />
        </div>
        <div className="mt-4 p-3 bg-slate-100 rounded-md">
            <p className="font-mono text-sm text-slate-800">
                {steps[currentStep]?.message || 'Visualization not started.'}
            </p>
        </div>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">Pseudocode</h2>
        <CodeBlock code={alphaBetaCode} />
      </Card>
    </div>
  );
};

export default AlphaBetaSection;
