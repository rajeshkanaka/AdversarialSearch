
import React from 'react';
import Card from '../ui/Card';

const IntroductionSection: React.FC = () => {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Welcome to Interactive Adversarial Search</h1>
        <p className="text-xl text-gray-600">Explore game-playing algorithms from Chapter 5 of "Artificial Intelligence: A Modern Approach".</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">What is Adversarial Search?</h2>
        <p className="text-gray-700 leading-relaxed">
          Adversarial search problems, often known as <strong className="font-semibold">games</strong>, arise in competitive environments where multiple agents with conflicting goals interact. In these scenarios, an agent's success depends not only on its own actions but also on the actions of other agents.
        </p>
        <p className="mt-4 text-gray-700 leading-relaxed">
          This module focuses on a specific type: deterministic, turn-taking, two-player, zero-sum games with perfect information (like chess or Go). In these games, one player's gain is exactly the other player's loss. Our goal is to find an optimal strategy to make the best possible move.
        </p>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">How to Use This App</h2>
        <p className="text-gray-700 leading-relaxed">
          Use the sidebar to navigate through the key concepts of the chapter. Each section provides:
        </p>
        <ul className="list-disc list-inside mt-4 space-y-2 text-gray-700">
          <li>A concise explanation of the core idea.</li>
          <li>An <strong className="font-semibold">interactive visualization</strong> that lets you see the algorithm in action, step-by-step.</li>
          <li>A code snippet demonstrating a practical implementation.</li>
        </ul>
        <p className="mt-4 text-gray-700 leading-relaxed">
          Start by exploring the <strong className="font-semibold">Minimax algorithm</strong>, the fundamental method for making optimal decisions in these games.
        </p>
      </Card>
    </div>
  );
};

export default IntroductionSection;
