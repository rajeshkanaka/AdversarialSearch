
import React from 'react';
import Card from '../ui/Card';

const StateOfTheArtSection: React.FC = () => {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">State-of-the-Art Game Programs</h1>
        <p className="text-xl text-slate-600">A look at how these algorithms power world-class game-playing AI.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">From Theory to World Champions</h2>
        <p className="text-slate-700 leading-relaxed">
            The algorithms and concepts in this chapter are not just theoretical; they form the foundation of programs that have surpassed the best human players in many games. Progress has come from both algorithmic improvements and massive increases in computational power.
        </p>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-xl font-semibold mb-3">Chess</h3>
          <p className="text-slate-700">
            IBM's <strong className="font-semibold">Deep Blue</strong> famously defeated world champion Garry Kasparov in 1997. It used custom hardware to perform an enormous Alpha-Beta search, reaching depths of 14-ply routinely, with singular extensions going as deep as 40-ply. Modern programs like <strong className="font-semibold">Stockfish</strong> run on commodity hardware and are even stronger, thanks to algorithmic improvements like null-move heuristics and futility pruning.
          </p>
        </Card>
        <Card>
          <h3 className="text-xl font-semibold mb-3">Checkers</h3>
          <p className="text-slate-700">
            The game of checkers was <strong className="font-semibold">solved</strong> in 2007 by Jonathan Schaeffer's <strong className="font-semibold">Chinook</strong> program. Using a combination of Alpha-Beta search and a massive endgame database (covering all positions with 10 or fewer pieces), it proved that with perfect play, the game always results in a draw.
          </p>
        </Card>
        <Card>
          <h3 className="text-xl font-semibold mb-3">Backgammon</h3>
          <p className="text-slate-700">
            As a stochastic game, Backgammon requires a different approach. Gerry Tesauro's <strong className="font-semibold">TD-Gammon</strong> used a neural network trained through reinforcement learning (by playing millions of games against itself) to create a powerful evaluation function. This allowed it to play at a world-champion level with a very shallow Expectiminimax search (2 or 3-ply).
          </p>
        </Card>
        <Card>
          <h3 className="text-xl font-semibold mb-3">Go</h3>
          <p className="text-slate-700">
            Go has a massive branching factor, making traditional Alpha-Beta search ineffective. The breakthrough came with <strong className="font-semibold">AlphaGo</strong>, which combined neural networks with Monte Carlo Tree Search (MCTS). MCTS is a probabilistic search method that balances exploring new moves with exploiting promising ones, allowing it to navigate the enormous search space effectively.
          </p>
        </Card>
      </div>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Key Takeaways</h2>
        <ul className="list-disc list-inside mt-2 space-y-2 text-slate-700">
            <li>There is no one-size-fits-all algorithm; the best approach depends on the game's characteristics (branching factor, deterministic vs. stochastic, perfect vs. imperfect information).</li>
            <li>For many classic games, the combination of a deep <strong className="font-semibold">Alpha-Beta search</strong> and a finely tuned <strong className="font-semibold">heuristic evaluation function</strong> is the path to high performance.</li>
            <li>For games with huge search spaces or stochastic elements, modern approaches often blend search with <strong className="font-semibold">machine learning</strong> to create better heuristics or guide the search process itself.</li>
        </ul>
      </Card>
    </div>
  );
};

export default StateOfTheArtSection;
