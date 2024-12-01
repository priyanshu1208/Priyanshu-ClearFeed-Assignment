async function submitQuery() {
    document.getElementById('links').value = '';
    document.getElementById('answer').value = '';
    const query_question = document.getElementById('query_question').value;
    const selected_model = document.getElementById('model_dropdown').value;
    const model_number = selected_model;

    const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_question, model_number })
    });

    const data = await response.json();
    document.getElementById('links').value = data.links.join('\n');
    document.getElementById('answer').value = data.answer;
}