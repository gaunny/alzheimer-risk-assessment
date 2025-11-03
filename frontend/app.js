document.getElementById('assessmentForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const patientData = {
        subject_id: formData.get('subject_id'),
        age: parseInt(formData.get('age')),
        sex: formData.get('sex'),
        mmse: formData.get('mmse') ? parseFloat(formData.get('mmse')) : null,
        feature_values: JSON.parse(formData.get('feature_values'))
    };

    try {
        const response = await fetch('http://localhost:8000/assess-risk/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        document.getElementById('result').innerHTML = `
            <h3>Error</h3>
            <p>${error.message}</p>
        `;
    }
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <h2>Assessment Result for ${result.subject_id}</h2>
        <p><strong>Risk Level:</strong> ${result.risk_level}</p>
        <p><strong>Risk Score:</strong> ${result.risk_score ? result.risk_score.toFixed(2) : 'N/A'}</p>

        <h3>Clinical Explanation</h3>
        <div style="white-space: pre-wrap;">${result.explanation}</div>

        <h3>Cited Literature</h3>
        <ul>
            ${result.cited_literature.map(ref => `<li>${ref}</li>`).join('')}
        </ul>
    `;
}