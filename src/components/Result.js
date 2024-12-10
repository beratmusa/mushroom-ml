import React from "react";

function Result({ prediction, probabilities }) {
  if (!prediction) return null;

  return (
    <div>
      <h2>Prediction Results</h2>
      <p>Predicted Class: {prediction}</p>
      <p>Probabilities:</p>
      <ul>
        {probabilities.map((prob, index) => (
          <li key={index}>
            Class {index}: {prob.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Result;
