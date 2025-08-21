import { useState } from 'react'

export default function InputText() {
    const [inputTxt, setInputTxt] = useState([""])
    const [summary, setSummary] = useState([""])
    
    const onSubmit = async (e) => {
        e.preventDefault();

        const response = await fetch("http://127.0.0.1:5000/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "text": inputTxt })    
        });
        const data = await response.json()
        setSummary(data.summary)
        console.log(data.summary)
    }

    return (
        <form onSubmit={onSubmit}>
            {summary && (
                <div>
                    <p>{summary}</p>
                </div>
            )}
            <label htmlFor='inputText'>Input Text</label><br />
            <textarea id='inputText' onChange={e => setInputTxt(e.target.value)} required></textarea><br />
            <button type="Submit">Summarize</button>
        </form>
    )
}
