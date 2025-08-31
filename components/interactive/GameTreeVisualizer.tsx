
import React from 'react';
import { VisualTreeNode } from '../../types';

interface GameTreeVisualizerProps {
  tree: VisualTreeNode | null;
  showAlphaBeta?: boolean;
}

const Node: React.FC<{ node: VisualTreeNode; level: number; showAlphaBeta?: boolean; }> = ({ node, level, showAlphaBeta }) => {
    const isMaxNode = level % 2 === 0;
    const nodeTypeColor = node.isLeaf ? 'bg-gray-500' : isMaxNode ? 'bg-green-500' : 'bg-red-500';
    const nodeBorderColor = node.isCurrent ? 'border-indigo-500' : 'border-gray-300';
    const textColor = node.isPruned ? 'text-gray-400' : 'text-white';

    const valueDisplay = node.calculatedValue !== undefined ? node.calculatedValue : node.isLeaf ? node.value : '?';

  return (
    <div
      className={`absolute transition-all duration-300 ${node.isPruned ? 'opacity-30' : 'opacity-100'}`}
      style={{ left: `${node.x}px`, top: `${node.y}px`, transform: 'translate(-50%, -50%)' }}
    >
      <div className={`relative w-20 h-14 rounded-lg shadow-md flex flex-col items-center justify-center border-2 ${nodeBorderColor} ${node.isPath ? 'ring-2 ring-yellow-400' : ''}`}>
        <div className={`absolute -top-3 left-1/2 -translate-x-1/2 px-2 py-0.5 text-xs font-semibold rounded-full ${nodeTypeColor} text-white`}>
            {node.isLeaf ? 'LEAF' : isMaxNode ? 'MAX' : 'MIN'}
        </div>
        
        <div className="text-lg font-bold text-gray-800">
          {valueDisplay}
        </div>
         <div className="text-xs text-gray-500">{node.id}</div>
      </div>
       {showAlphaBeta && !node.isLeaf && (
         <div className="absolute top-14 w-full text-center text-xs font-mono mt-1">
          <span className="text-green-600">α:{node.alpha ?? '-∞'}</span> <span className="text-red-600">β:{node.beta ?? '+∞'}</span>
        </div>
      )}
    </div>
  );
};

const Edge: React.FC<{ from: VisualTreeNode; to: VisualTreeNode; isPruned?: boolean; isPath?: boolean; }> = ({ from, to, isPruned, isPath }) => {
  const strokeColor = isPruned ? '#d1d5db' : isPath ? '#facc15' : '#9ca3af';
  const strokeWidth = isPath ? '3' : '2';

  return (
    <line
      x1={from.x}
      y1={from.y + 28}
      x2={to.x}
      y2={to.y - 28}
      stroke={strokeColor}
      strokeWidth={strokeWidth}
      className="transition-all duration-300"
    />
  );
};

const renderTreeRecursive = (node: VisualTreeNode, level: number, showAlphaBeta?: boolean): { nodes: JSX.Element[]; edges: JSX.Element[] } => {
  let nodes: JSX.Element[] = [<Node key={node.id} node={node} level={level} showAlphaBeta={showAlphaBeta} />];
  let edges: JSX.Element[] = [];

  if (node.children) {
    node.children.forEach(child => {
      edges.push(<Edge key={`${node.id}-${child.id}`} from={node} to={child} isPruned={child.isPruned} isPath={child.isPath && node.isPath} />);
      const childResult = renderTreeRecursive(child, level + 1, showAlphaBeta);
      nodes = nodes.concat(childResult.nodes);
      edges = edges.concat(childResult.edges);
    });
  }
  
  return { nodes, edges };
};

const GameTreeVisualizer: React.FC<GameTreeVisualizerProps> = ({ tree, showAlphaBeta = false }) => {
  if (!tree) return <div className="text-center text-slate-500">Loading tree...</div>;
  
  const { nodes, edges } = renderTreeRecursive(tree, 0, showAlphaBeta);

  return (
    <div className="relative w-full h-full">
      <svg width="100%" height="320px" className="absolute top-0 left-0">
        <g>{edges}</g>
      </svg>
      {nodes}
    </div>
  );
};

export default GameTreeVisualizer;
