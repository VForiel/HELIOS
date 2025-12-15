import { createContext, useContext } from 'react';

export const PipelineContext = createContext({
    onInspect: () => { },
});

export const usePipelineContext = () => useContext(PipelineContext);
