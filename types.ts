
export enum SectionId {
  Introduction = 'introduction',
  Minimax = 'minimax',
  MultiplayerGames = 'multiplayer-games',
  AlphaBeta = 'alpha-beta',
  ImperfectDecisions = 'imperfect-decisions',
  StochasticGames = 'stochastic-games',
  StateOfTheArt = 'state-of-the-art',
}

export interface GameTreeNode {
  id: string;
  value?: number;
  children?: GameTreeNode[];
}

export interface VisualTreeNode extends GameTreeNode {
  x: number;
  y: number;
  isLeaf: boolean;
  children?: VisualTreeNode[];
  // State for visualization
  calculatedValue?: number;
  alpha?: number | string;
  beta?: number | string;
  isPruned?: boolean;
  isCurrent?: boolean;
  isPath?: boolean;
}
