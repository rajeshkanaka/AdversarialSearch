
import { useState, useCallback, useEffect } from 'react';
import { GameTreeNode, VisualTreeNode } from '../types';

interface Step {
  tree: VisualTreeNode;
  message: string;
}

const prepareTree = (node: GameTreeNode, depth = 0, x = 400, width = 800): VisualTreeNode => {
  const isLeaf = !node.children || node.children.length === 0;

  const children: VisualTreeNode[] | undefined = node.children
    ? node.children.map((child, index) => {
        const childWidth = width / node.children.length;
        return prepareTree(
          child,
          depth + 1,
          x - width / 2 + childWidth / 2 + index * childWidth,
          childWidth
        );
      })
    : undefined;

  const visualNode: VisualTreeNode = {
    ...node,
    x,
    y: 40 + depth * 90,
    isLeaf,
    alpha: '-∞',
    beta: '+∞',
    children,
  };
  return visualNode;
};

const cloneTree = (node: VisualTreeNode): VisualTreeNode => JSON.parse(JSON.stringify(node));

export const useAlphaBeta = (initialTree: GameTreeNode) => {
  const [initialVisualTree] = useState(() => prepareTree(initialTree));
  const [tree, setTree] = useState<VisualTreeNode>(initialVisualTree);
  const [steps, setSteps] = useState<Step[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  const buildSteps = useCallback(() => {
    const localSteps: Step[] = [];
    let currentTree = cloneTree(initialVisualTree);

    const addStep = (message: string) => {
      localSteps.push({ tree: cloneTree(currentTree), message });
    };
    
    const findNode = (node: VisualTreeNode, id: string): VisualTreeNode | null => {
        if (node.id === id) return node;
        if (node.children) {
            for (const child of node.children) {
                const found = findNode(child, id);
                if (found) return found;
            }
        }
        return null;
    }
    
    const pruneChildren = (node: VisualTreeNode) => {
        if (!node.children) return;
        let childrenToPrune = node.children.filter(c => c.calculatedValue === undefined);
        for (const child of childrenToPrune) {
            child.isPruned = true;
            addStep(`Pruning node ${child.id} and its descendants.`);
            pruneChildren(child);
        }
    }

    function alphaBeta(nodeId: string, depth: number, alpha: number, beta: number, maximizingPlayer: boolean): number {
        const node = findNode(currentTree, nodeId)!;
        node.isCurrent = true;
        node.alpha = alpha === -Infinity ? '-∞' : alpha;
        node.beta = beta === Infinity ? '+∞' : beta;

        if (node.isLeaf) {
            addStep(`Terminal node ${node.id}. Utility is ${node.value}.`);
            node.calculatedValue = node.value;
            node.isCurrent = false;
            return node.value!;
        }

        if (maximizingPlayer) {
            let value = -Infinity;
            addStep(`MAX node ${node.id} [α=${node.alpha}, β=${node.beta}]. Initializing value to -Infinity.`);
            for (const child of node.children!) {
                const childResult = alphaBeta(child.id, depth + 1, alpha, beta, false);
                value = Math.max(value, childResult);
                const oldAlpha = alpha;
                alpha = Math.max(alpha, value);
                
                if (alpha > oldAlpha) {
                    node.alpha = alpha;
                    addStep(`MAX node ${node.id}: received ${childResult}. value=${value}. α updated to ${alpha}.`);
                } else {
                    addStep(`MAX node ${node.id}: received ${childResult}. value=${value}, α=${alpha}.`);
                }

                if (alpha >= beta) {
                    addStep(`PRUNING! At MAX node ${node.id}, α(${alpha}) >= β(${beta}).`);
                    pruneChildren(node);
                    break;
                }
            }
            node.calculatedValue = value;
            addStep(`MAX node ${node.id} finished. Backing up value: ${value}.`);
            node.isCurrent = false;
            return value;
        } else {
            let value = +Infinity;
            addStep(`MIN node ${node.id} [α=${node.alpha}, β=${node.beta}]. Initializing value to +Infinity.`);
            for (const child of node.children!) {
                const childResult = alphaBeta(child.id, depth + 1, alpha, beta, true);
                value = Math.min(value, childResult);
                const oldBeta = beta;
                beta = Math.min(beta, value);

                if (beta < oldBeta) {
                    node.beta = beta;
                    addStep(`MIN node ${node.id}: received ${childResult}. value=${value}. β updated to ${beta}.`);
                } else {
                     addStep(`MIN node ${node.id}: received ${childResult}. value=${value}, β=${beta}.`);
                }

                if (beta <= alpha) {
                    addStep(`PRUNING! At MIN node ${node.id}, β(${beta}) <= α(${alpha}).`);
                    pruneChildren(node);
                    break;
                }
            }
            node.calculatedValue = value;
            addStep(`MIN node ${node.id} finished. Backing up value: ${value}.`);
            node.isCurrent = false;
            return value;
        }
    }
    
    addStep("Starting Alpha-Beta Pruning calculation at the root.");
    alphaBeta(currentTree.id, 0, -Infinity, Infinity, true);
    addStep("Alpha-Beta Pruning complete. The root value is the optimal score.");
    setSteps(localSteps);
    setTree(localSteps[0].tree);
  }, [initialVisualTree]);
  
  useEffect(() => {
    buildSteps();
  }, [buildSteps]);
  
  useEffect(() => {
    if (isRunning && currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
        setTree(steps[currentStep + 1].tree);
      }, 700);
      return () => clearTimeout(timer);
    } else {
      setIsRunning(false);
    }
  }, [isRunning, currentStep, steps]);

  const run = () => {
    if (steps.length > 0) {
      reset();
      setIsRunning(true);
    }
  };

  const reset = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setTree(steps.length > 0 ? steps[0].tree : initialVisualTree);
  };
  
  return { tree, steps, currentStep, run, reset, isRunning };
};
