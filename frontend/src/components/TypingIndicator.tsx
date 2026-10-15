import React from "react";

/** Three bouncing dots, like a chat app's "typing..." indicator. Purely
 * decorative -- pair with LoadingStatus for the actual status text. */
export const TypingIndicator: React.FC = () => {
  return (
    <span className="typing-dots" role="presentation" aria-hidden="true">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </span>
  );
};
