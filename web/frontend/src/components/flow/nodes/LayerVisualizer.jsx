import React, { useState } from 'react';
import { Eye, X } from 'lucide-react';

export default function LayerVisualizer({ type, config }) {
    const [showPreview, setShowPreview] = useState(false);
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePreview = async () => {
        setShowPreview(true);
        if (image) return; // Cached if nothing changed (warning: if config changed, we don't know easily unless we effect)
        // For simplicity, re-fetch always on open to ensure freshness

        setLoading(true);
        try {
            const payload = { type, config };
            const response = await fetch('/api/preview_layer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const blob = await response.blob();
                setImage(URL.createObjectURL(blob));
            } else {
                console.error("Preview failed");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <button
                onClick={handlePreview}
                className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                title="Visualize Layer"
            >
                <Eye className="w-4 h-4" />
            </button>

            {showPreview && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setShowPreview(false)}>
                    <div className="bg-white dark:bg-slate-900 rounded-lg shadow-2xl border border-slate-200 dark:border-slate-700 max-w-2xl w-full flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="flex items-center justify-between p-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
                            <h3 className="font-semibold text-slate-800 dark:text-slate-200 capitalize">{type} Visualization</h3>
                            <button onClick={() => setShowPreview(false)} className="text-slate-500 hover:text-slate-700 dark:hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="p-4 flex items-center justify-center min-h-[300px] bg-slate-100 dark:bg-black/20">
                            {loading ? (
                                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                            ) : image ? (
                                <img src={image} alt="Preview" className="max-h-[600px] object-contain rounded shadow-sm" />
                            ) : (
                                <div className="text-slate-500">Failed to load preview</div>
                            )}
                        </div>
                        <div className="p-3 border-t border-slate-200 dark:border-slate-800 flex justify-end">
                            <button
                                onClick={handlePreview}
                                className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-500 dark:hover:text-blue-300 mr-auto font-medium"
                            >
                                Refresh
                            </button>
                            <button onClick={() => setShowPreview(false)} className="px-4 py-1.5 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 rounded text-sm text-slate-700 dark:text-slate-300 transition-colors">
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
