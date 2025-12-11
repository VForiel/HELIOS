import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

const pythonKeywords = [
    'import', 'from', 'as', 'def', 'return', 'if', 'else', 'elif', 
    'for', 'in', 'class', 'try', 'except', 'while', 'with', 'pass', 
    'break', 'continue', 'lambda', 'True', 'False', 'None'
];

const highlightCode = (code) => {
    if (!code) return [];

    // Simple tokenizer
    // We will split by lines first
    const lines = code.split('\n');
    
    return lines.map((line, lineIdx) => {
        // Process each line
        // 1. Comments
        const commentSplit = line.split('#');
        let codePart = commentSplit[0];
        const commentPart = commentSplit.slice(1).join('#');

        // 2. Strings (simple approximation, doesn't handle multiline strings correctly in this simple view)
        // We'll use a regex to find strings and replace them with placeholders, then restore
        const tokens = [];
        let current = codePart;
        
        // Regex for strings: "..." or '...'
        const stringRegex = /(".*?"|'.*?')/g;
        const parts = current.split(stringRegex);
        
        const highlightedParts = parts.map((part, i) => {
            if (part.match(stringRegex)) {
                return <span key={i} className="text-green-600 dark:text-green-400">{part}</span>;
            }
            
            // Process keywords and numbers in non-string parts
            // Split by word boundaries but keep delimiters
            const words = part.split(/(\b|\W)/);
            return words.map((word, wIdx) => {
                if (pythonKeywords.includes(word)) {
                    return <span key={`${i}-${wIdx}`} className="text-purple-600 dark:text-purple-400 font-semibold">{word}</span>;
                } else if (!isNaN(parseFloat(word)) && isFinite(word)) {
                    return <span key={`${i}-${wIdx}`} className="text-orange-600 dark:text-orange-400">{word}</span>;
                } else if (word.match(/^[A-Z][a-zA-Z0-9_]*$/)) {
                     // Likely a class
                     return <span key={`${i}-${wIdx}`} className="text-yellow-600 dark:text-yellow-400">{word}</span>;
                } else if (word.match(/^[a-zA-Z_][a-zA-Z0-9_]*\(/)) {
                    // Function call (heuristic: word followed by open paren, but we split by \W so paren is separate)
                    // This simple split makes lookahead hard. Let's ignore function colors for now or try to match functions.
                    return word;
                }
                return word;
            });
        });

        return (
            <div key={lineIdx} className="whitespace-pre font-mono text-sm leading-5">
                {highlightedParts}
                {commentSplit.length > 1 && (
                    <span className="text-slate-500 italic">#{commentPart}</span>
                )}
            </div>
        );
    });
};

export default function CodeViewer({ code }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    return (
        <div className="relative w-full h-full flex flex-col bg-slate-50 dark:bg-slate-900 rounded-md overflow-hidden border border-slate-200 dark:border-slate-700">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-100 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">python</span>
                <button
                    onClick={handleCopy}
                    className="flex items-center gap-2 px-3 py-1 text-xs font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
                >
                    {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
                    {copied ? 'Copied!' : 'Copy'}
                </button>
            </div>

            {/* Code Area */}
            <div className="flex-1 overflow-auto p-4 custom-scrollbar select-text cursor-text">
                {highlightCode(code)}
            </div>
        </div>
    );
}
