import React from 'react';
import { Minus, Plus } from 'lucide-react';

export default function NumberInput({
    value,
    onChange,
    min,
    max,
    step = 1,
    className = "",
    size = "sm",
    disabled = false
}) {
    // Handle internal change
    const handleChange = (e) => {
        let val = parseFloat(e.target.value);
        if (isNaN(val)) val = "";
        triggerChange(val);
    };

    const triggerChange = (val) => {
        if (val !== "" && min !== undefined && val < min) val = min;
        if (val !== "" && max !== undefined && val > max) val = max;
        onChange(val);
    };

    const increment = () => {
        let current = parseFloat(value) || 0;
        let next = current + Number(step);
        // Precision fix for floats
        if (!Number.isInteger(step)) {
            next = parseFloat(next.toFixed(10));
        }
        triggerChange(next);
    };

    const decrement = () => {
        let current = parseFloat(value) || 0;
        let next = current - Number(step);
        if (!Number.isInteger(step)) {
            next = parseFloat(next.toFixed(10));
        }
        triggerChange(next);
    };

    // Styling
    const baseClasses = "flex items-center border border-slate-300 dark:border-slate-700 rounded-md overflow-hidden bg-white dark:bg-slate-800 transition-colors focus-within:ring-2 focus-within:ring-blue-500/50 focus-within:border-blue-500";
    const btnClasses = "flex items-center justify-center bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 active:bg-slate-300 dark:active:bg-slate-500 text-slate-600 dark:text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
    const inputClasses = "w-full text-center bg-transparent border-none outline-none text-slate-900 dark:text-slate-100 font-mono focus:ring-0 p-0";

    // Sizes
    const sizes = {
        xs: { h: "h-6", icon: "w-3 h-3", text: "text-xs", pad: "px-1", w: "w-16" },
        sm: { h: "h-8", icon: "w-3.5 h-3.5", text: "text-sm", pad: "px-2", w: "w-24" },
        md: { h: "h-10", icon: "w-4 h-4", text: "text-base", pad: "px-3", w: "w-32" }
    };

    const s = sizes[size] || sizes.sm;

    return (
        <div className={`${baseClasses} ${s.h} ${className}`}>
            <button
                type="button"
                onClick={decrement}
                disabled={disabled || (min !== undefined && value <= min)}
                className={`${btnClasses} h-full aspect-square`}
                tabIndex={-1}
            >
                <Minus className={s.icon} />
            </button>
            <div className="flex-1 min-w-[2rem] h-full flex items-center bg-transparent relative hover:bg-slate-50 dark:hover:bg-slate-800/50">
                <input
                    type="number"
                    value={value}
                    onChange={handleChange}
                    min={min}
                    max={max}
                    step={step}
                    disabled={disabled}
                    className={`${inputClasses} ${s.text} h-full`}
                />
                {/* Hide Arrows via CSS in global or style tag */}
                <style jsx>{`
                    input[type=number]::-webkit-inner-spin-button, 
                    input[type=number]::-webkit-outer-spin-button { 
                        -webkit-appearance: none; 
                        margin: 0; 
                    }
                    input[type=number] {
                        -moz-appearance: textfield;
                    }
                `}</style>
            </div>
            <button
                type="button"
                onClick={increment}
                disabled={disabled || (max !== undefined && value >= max)}
                className={`${btnClasses} h-full aspect-square`}
                tabIndex={-1}
            >
                <Plus className={s.icon} />
            </button>
        </div>
    );
}
