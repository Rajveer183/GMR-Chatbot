import React from "react";

interface SuggestedQuestionsProps {
  questions: string[];
  onPick: (q: string, index: number) => void;
}

export const SuggestedQuestions: React.FC<SuggestedQuestionsProps> = ({ questions, onPick }) => {
  return (
    <div className="qa-chips">
      {questions.map((q, idx) => (
        <button key={idx} type="button" className="qa-chip" onClick={() => onPick(q, idx)}>
          {q}
        </button>
      ))}
    </div>
  );
};
