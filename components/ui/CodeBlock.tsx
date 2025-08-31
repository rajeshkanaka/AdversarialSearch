
import React from 'react';

interface CodeBlockProps {
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code }) => {
  return (
    <pre className="bg-gray-800 text-white font-mono text-sm p-4 rounded-md overflow-x-auto">
      <code>{code.trim()}</code>
    </pre>
  );
};

export default CodeBlock;
