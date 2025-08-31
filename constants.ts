
import { GameTreeNode, SectionId } from './types';

export const GAME_TREE_DATA: GameTreeNode = {
  id: 'A',
  children: [
    {
      id: 'B',
      children: [{ id: 'B1', value: 3 }, { id: 'B2', value: 12 }, { id: 'B3', value: 8 }],
    },
    {
      id: 'C',
      children: [{ id: 'C1', value: 2 }, { id: 'C2', value: 4 }, { id: 'C3', value: 6 }],
    },
    {
      id: 'D',
      children: [{ id: 'D1', value: 14 }, { id: 'D2', value: 5 }, { id: 'D3', value: 2 }],
    },
  ],
};

export const SECTIONS = [
    { 
        id: 'group-intro',
        title: 'Fundamentals',
        children: [
            { id: SectionId.Introduction, title: '5.1 Games & Adversarial Search' },
            { id: SectionId.Minimax, title: '5.2 Optimal Decisions & Minimax' },
        ]
    },
    {
        id: 'group-algorithms',
        title: 'Core Algorithms',
        children: [
            { id: SectionId.AlphaBeta, title: '5.3 Alpha-Beta Pruning' },
        ]
    },
    {
        id: 'group-advanced',
        title: 'Advanced Concepts',
        children: [
            { id: SectionId.MultiplayerGames, title: '5.2.2 Multiplayer Games' },
            { id: SectionId.ImperfectDecisions, title: '5.4 Imperfect Decisions' },
            { id: SectionId.StochasticGames, title: '5.5 Stochastic Games' },
        ]
    },
    {
        id: 'group-conclusion',
        title: 'Conclusion',
        children: [
            { id: SectionId.StateOfTheArt, title: '5.7 State-of-the-Art Programs' },
        ]
    }
];
