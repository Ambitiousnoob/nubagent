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

    const statusClass = isRunning ? "active" : status;

    const getPrefix = () => {
        if (isRunning) return `[${spinnerFrames[frame]} ACTIVE]`;
        switch (status) {
            case "success": return "[   OK   ]";
            case "error": return "[ FATAL  ]";
            case "think": return "[ THINK  ]";
            case "warn": return "[  WARN  ]";
            case "info": return "[  INFO  ]";
            case "fetch": return "[  CURL  ]";
            default: return ">";
        }
    };

    return (
        <div className={`tm-row tm-row--${statusClass}`}>
            <div className={`tm-prefix tm-prefix--${statusClass}`}>
                {getPrefix()}
            </div>
            <div className={`tm-text tm-text--${statusClass}`}>
                {text}
            </div>
        </div>
    );
}
