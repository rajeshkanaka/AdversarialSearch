
import React from 'react';
import Card from '../ui/Card';

const MultiplayerGamesSection: React.FC = () => {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Optimal Decisions in Multiplayer Games</h1>
        <p className="text-xl text-slate-600">Extending Minimax beyond two-player, zero-sum scenarios.</p>
      </header>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Beyond Two Players</h2>
        <p className="text-slate-700 leading-relaxed">
          Many popular games, like Diplomacy or Settlers of Catan, involve more than two players. The standard Minimax algorithm doesn't directly apply because the concept of a single opponent to minimize our score is no longer valid.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          To handle this, we need to replace the single utility value at each node with a <strong className="font-semibold">vector of utility values</strong>, one for each player. For a three-player game (A, B, C), a terminal node would have a vector like `(vA, vB, vC)` representing the outcome for each player.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          When backing up values, a player at a given node `n` will choose the move that leads to a successor state with the highest utility <strong className="font-semibold">for themselves</strong>.
        </p>
      </Card>

      <Card>
        <h2 className="text-2xl font-semibold mb-4">Example: Three-Player Game Tree</h2>
        <p className="text-slate-700 mb-4">
          This diagram, based on Figure 5.4 from the book, shows a game tree with three players (A, B, C). Each node is labeled with a utility vector `(vA, vB, vC)`. The backed-up value of a node is the utility vector of the successor state that the current player would choose.
        </p>
        <div className="flex justify-center p-4 border rounded-lg bg-slate-50">
          <svg width="450" height="250" viewBox="0 0 500 270" className="font-sans text-sm">
            {/* <!-- Player Labels --> */}
            <text x="20" y="35" className="font-bold">A</text>
            <text x="20" y="95" className="font-bold">B</text>
            <text x="20" y="155" className="font-bold">C</text>
            <text x="20" y="215" className="font-bold">A</text>

            {/* <!-- Level 0 (A) --> */}
            <circle cx="250" cy="30" r="15" fill="#a7f3d0" stroke="#10b981" strokeWidth="2" />
            <text x="250" y="35" textAnchor="middle" className="font-bold">(1,2,6)</text>

            {/* <!-- Level 1 (B) --> */}
            <line x1="250" y1="45" x2="150" y2="75" stroke="#9ca3af" strokeWidth="2" />
            <line x1="250" y1="45" x2="350" y2="75" stroke="#9ca3af" strokeWidth="2" />
            <circle cx="150" cy="90" r="15" fill="#fecaca" stroke="#ef4444" strokeWidth="2" />
            <text x="150" y="95" textAnchor="middle" className="font-bold">(1,2,6)</text>
            <circle cx="350" cy="90" r="15" fill="#fecaca" stroke="#ef4444" strokeWidth="2" />
            <text x="350" y="95" textAnchor="middle" className="font-bold">(1,5,2)</text>

            {/* <!-- Level 2 (C) --> */}
            <line x1="150" y1="105" x2="100" y2="135" stroke="#9ca3af" strokeWidth="2" />
            <line x1="150" y1="105" x2="200" y2="135" stroke="#9ca3af" strokeWidth="2" />
            <circle cx="100" cy="150" r="15" fill="#bfdbfe" stroke="#3b82f6" strokeWidth="2" />
            <text x="100" y="155" textAnchor="middle" className="font-bold">(1,2,6)</text>
            <circle cx="200" cy="150" r="15" fill="#bfdbfe" stroke="#3b82f6" strokeWidth="2" />
            <text x="200" y="155" textAnchor="middle" className="font-bold">(6,1,2)</text>

            {/* <!-- Terminal Nodes --> */}
            <line x1="100" y1="165" x2="75" y2="195" stroke="#9ca3af" strokeWidth="2" />
            <line x1="100" y1="165" x2="125" y2="195" stroke="#9ca3af" strokeWidth="2" />
            <rect x="55" y="210" width="40" height="20" fill="#e5e7eb" />
            <text x="75" y="224" textAnchor="middle">(1,2,6)</text>
            <rect x="105" y="210" width="40" height="20" fill="#e5e7eb" />
            <text x="125" y="224" textAnchor="middle">(4,2,3)</text>
            <text x="100" y="185" className="text-xs">C chooses left (6>3)</text>

            {/* <!-- Explanation --> */}
            <text x="250" y="260" textAnchor="middle" className="text-slate-600">At node X, it's C's turn. C chooses the left move because its utility (6) is greater than its utility on the right (2).</text>
          </svg>
        </div>
      </Card>
      
      <Card>
        <h2 className="text-2xl font-semibold mb-4">Alliances and Complications</h2>
        <p className="text-slate-700 leading-relaxed">
          Multiplayer games introduce complex social dynamics, most notably <strong className="font-semibold">alliances</strong>. Players may form temporary coalitions to oppose a stronger player. For example, if player C is in a dominant position, it might be optimal for both A and B to attack C rather than each other.
        </p>
        <p className="mt-4 text-slate-700 leading-relaxed">
          This collaboration can emerge from purely selfish behavior. However, as soon as the balance of power shifts, the alliance may break. This makes predicting outcomes much harder and moves game theory into the realm of social and economic modeling.
        </p>
      </Card>
    </div>
  );
};

export default MultiplayerGamesSection;
