import React from "react";

interface InsightListProps {
  title: string;
  items: string[];
}

export const InsightList: React.FC<InsightListProps> = ({ title, items }) => {
  if (!items?.length) return null;
  return (
    <section className="chat-result-section">
      <h4 className="chat-section-heading">{title}</h4>
      <ul className="chat-insight-list">
        {items.map((item, idx) => (
          <li key={idx}>{item}</li>
        ))}
      </ul>
    </section>
  );
};
