
import React from 'react';
import Card from '../ui/Card';
import Button from '../ui/Button';
import CodeBlock from '../ui/CodeBlock';
import GameTreeVisualizer from '../interactive/GameTreeVisualizer';
import { useMinimax } from '../../hooks/useMinimax';
import { GAME_TREE_DATA } from '../../constants';

const minimaxCode = `
function minimax(node, depth, maximizingPlayer) {
  if (depth === 0 || node.isTerminal) {
    return node.utility;
  }

  if (maximizingPlayer) {
    let bestValue = -Infinity;
    for (let child of node.children) {
      const value = minimax(child, depth - 1, false);
      bestValue = Math.max(bestValue, value);
    }
    return bestValue;
  } else { // Minimizing player
    let bestValue = +Infinity;
    for (let child of node.children) {
      const value = minimax(child, depth - 1, true);
      bestValue = Math.min(bestValue, value);
    }
    return bestValue;
  }
}
`;

const MinimaxSection: React.FC = () => {
  const { tree, steps, currentStep, run, reset, isRunning } = useMinimax(GAME_TREE_DATA);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">The Minimax Algorithm</h1>
        <p className="text-xl text-slate-600">Finding the optimal move by assuming your opponent plays optimally too.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Core Concept</h2>
        <p className="text-slate-700 leading-relaxed">
          The Minimax algorithm computes the optimal strategy for a player in a zero-sum game. It works by recursively exploring the game tree. The core idea is to find the move that maximizes the utility for the current player (MAX), assuming the opponent (MIN) will always play to minimize MAX's utility.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          It performs a complete depth-first search of the game tree. At each level, it "backs up" the values from the leaf nodes:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-slate-700">
          <li><strong className="font-semibold text-green-600">MAX nodes</strong> choose the child with the highest value.</li>
          <li><strong className="font-semibold text-red-600">MIN nodes</strong> choose the child with the lowest value.</li>
        </ul>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-3">Interactive Visualization</h2>
        <p className="text-slate-600 mb-4">
          Press "Run" to watch the Minimax algorithm explore the game tree from Figure 5.2 in the book. The algorithm performs a post-order traversal to calculate and back up the utility values.
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
          <GameTreeVisualizer tree={tree} />
        </div>
         <div className="mt-4 p-3 bg-slate-100 rounded-md">
            <p className="font-mono text-sm text-slate-800">
                {steps[currentStep]?.message || 'Visualization not started.'}
            </p>
        </div>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">Pseudocode</h2>
        <CodeBlock code={minimaxCode} />
      </Card>
    </div>
  );
};

export default MinimaxSection;
