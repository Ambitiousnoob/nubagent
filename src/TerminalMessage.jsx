import React, { useState, useEffect } from "react";

const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export default function TerminalMessage({ status = "default", text, isRunning = false }) {
    const [frame, setFrame] = useState(0);

    useEffect(() => {
        if (!isRunning) return undefined;
        const interval = setInterval(() => {
            setFrame((prev) => (prev + 1) % spinnerFrames.length);
        }, 80);
        return () => clearInterval(interval);
    }, [isRunning]);

    const getPrefix = () => {
        if (isRunning) return <span className="text-cyan-400">[{spinnerFrames[frame]} ACTIVE]</span>;
        switch (status) {
            case "success": return <span className="text-green-400">[   OK   ]</span>;
            case "error":   return <span className="text-red-500">[ FATAL  ]</span>;
            case "warn":    return <span className="text-yellow-400">[  WARN  ]</span>;
            case "info":    return <span className="text-blue-400">[  INFO  ]</span>;
            case "fetch":   return <span className="text-purple-400">[  CURL  ]</span>;
            default:        return <span className="text-gray-500">❯</span>;
        }
    };

    return (
        <div className="font-mono text-sm mb-1 flex items-start space-x-3 bg-[#0d1117] p-1 rounded">
            <div className="flex-shrink-0 w-28 whitespace-pre font-bold select-none">
                {getPrefix()}
            </div>
            <div className={`break-words ${status === "error" ? "text-red-400" : "text-gray-300"}`}>
                {text}
            </div>
        </div>
    );
}
