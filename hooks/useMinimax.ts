
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
    children,
  };

  return visualNode;
};

const cloneTree = (node: VisualTreeNode): VisualTreeNode => JSON.parse(JSON.stringify(node));

export const useMinimax = (initialTree: GameTreeNode) => {
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

    function minimax(nodeId: string, depth: number, maximizingPlayer: boolean): number {
        const node = findNode(currentTree, nodeId)!;
        node.isCurrent = true;

        if (node.isLeaf) {
            addStep(`Evaluating terminal node ${node.id}. Utility is ${node.value}.`);
            node.calculatedValue = node.value;
            node.isCurrent = false;
            return node.value!;
        }

        if (maximizingPlayer) {
            let bestValue = -Infinity;
            addStep(`MAX node ${node.id}. Initializing bestValue to -Infinity.`);
            for (const child of node.children!) {
                const value = minimax(child.id, depth + 1, false);
                addStep(`MAX node ${node.id}: received ${value} from child ${child.id}. Current best is ${bestValue}. New best is ${Math.max(bestValue, value)}.`);
                bestValue = Math.max(bestValue, value);
            }
            node.calculatedValue = bestValue;
            addStep(`MAX node ${node.id} finished. Backing up value: ${bestValue}.`);
            node.isCurrent = false;
            return bestValue;
        } else {
            let bestValue = +Infinity;
            addStep(`MIN node ${node.id}. Initializing bestValue to +Infinity.`);
            for (const child of node.children!) {
                const value = minimax(child.id, depth + 1, true);
                addStep(`MIN node ${node.id}: received ${value} from child ${child.id}. Current best is ${bestValue}. New best is ${Math.min(bestValue, value)}.`);
                bestValue = Math.min(bestValue, value);
            }
            node.calculatedValue = bestValue;
            addStep(`MIN node ${node.id} finished. Backing up value: ${bestValue}.`);
            node.isCurrent = false;
            return bestValue;
        }
    }
    
    addStep("Starting Minimax calculation at the root.");
    minimax(currentTree.id, 0, true);
    addStep("Minimax complete. The root value is the optimal score.");
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
