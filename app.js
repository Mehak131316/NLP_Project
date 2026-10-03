const RESULTS_DIR = 'NLP_Assessment1-20261003T034029Z-1-001/NLP_Assessment1/results/';

// Global State
const state = {
    docsMap: [],
    docsStats: [],
    queries: [],
    retrieval: [],
    evaluation: [],
    isProvisional: false
};

// Helper: Parse CSV using PapaParse Promise
function loadCSV(filename) {
    return new Promise((resolve) => {
        Papa.parse(RESULTS_DIR + filename, {
            download: true,
            header: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (err) => {
                console.warn(`Could not load ${filename}:`, err);
                resolve([]);
            }
        });
    });
}

// Navigation Logic
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', (e) => {
        // Update active nav
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        const targetBtn = e.currentTarget;
        targetBtn.classList.add('active');
        
        // Update active section
        const targetId = targetBtn.getAttribute('data-target');
        document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
        document.getElementById(targetId).classList.add('active');
    });
});

// Render Table Helper
function renderTable(containerId, data, emptyMsg = "No data available") {
    const table = document.getElementById(containerId);
    if (!data || data.length === 0) {
        table.innerHTML = `<tr><td>${emptyMsg}</td></tr>`;
        return;
    }
    
    const headers = Object.keys(data[0]);
    
    // Thead
    let html = '<thead><tr>';
    headers.forEach(h => {
        html += `<th>${h}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Tbody
    data.forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
            let val = row[h] !== undefined ? row[h] : '';
            // Styling booleans or specifics
            let cellClass = '';
            let dispVal = val;
            
            if (val.toLowerCase && val.toLowerCase() === 'true') {
                cellClass = 'cell-true';
                dispVal = '✓ Yes';
            } else if (val.toLowerCase && val.toLowerCase() === 'false') {
                cellClass = 'cell-false';
                dispVal = '✗ No';
            } else if (val === '' || val === 'NaN') {
                cellClass = 'cell-null';
                dispVal = '-';
            }
            
            html += `<td class="${cellClass}">${dispVal}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

// Initialization
async function init() {
    try {
        // Load Tier 1 Datasets
        const [
            docStats, 
            prepResults, 
            tokenComp,
            stemLemma,
            posTagging,
            ner,
            ngramStats,
            bpe,
            retrieval,
            evaluation,
            pipelineComp
        ] = await Promise.all([
            loadCSV('document_statistics.csv'),
            loadCSV('preprocessing_results.csv'),
            loadCSV('tokenization_comparison.csv'),
            loadCSV('stemming_lemmatization.csv'),
            loadCSV('pos_tagging_results.csv'),
            loadCSV('ner_results.csv'),
            loadCSV('ngram_results.csv'),
            loadCSV('bpe_results.csv'),
            loadCSV('retrieval_results.csv'),
            loadCSV('evaluation_results.csv'),
            loadCSV('pipeline_comparison.csv')
        ]);

        // Hide loader
        document.getElementById('loader').classList.add('hidden');

        // Check Provisional status
        if (evaluation.length > 0) {
            const hasProvisional = evaluation.some(row => row['Relevance Source'] === 'PROVISIONAL');
            if (hasProvisional) {
                document.getElementById('provisional-banner').classList.remove('hidden');
                document.getElementById('provisional-banner').classList.add('show');
            }
        }

        // Section 2: Document Statistics
        if (docStats.length > 0) {
            const totalRow = docStats.find(r => r.doc_id === 'TOTAL');
            const onlyDocs = docStats.filter(r => r.doc_id !== 'TOTAL');
            
            if (totalRow) {
                const cardsHtml = `
                    <div class="stat-card">
                        <div class="label">Total Documents</div>
                        <div class="value">${onlyDocs.length}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Total Sentences</div>
                        <div class="value">${totalRow.sentences || 'N/A'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Total Tokens</div>
                        <div class="value">${totalRow.tokens || 'N/A'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Corpus Vocabulary</div>
                        <div class="value">${totalRow.vocabulary || 'N/A'}</div>
                    </div>
                `;
                document.getElementById('doc-summary-cards').innerHTML = cardsHtml;
            }
            renderTable('doc-stats-table', onlyDocs);
        }

        // Section 3: Tokenization
        if (tokenComp.length > 0) renderTable('token-comp-table', tokenComp);
        
        // Section 4: Preprocessing
        if (prepResults.length > 0) renderTable('prep-results-table', prepResults);
        if (stemLemma.length > 0) renderTable('stem-lemma-table', stemLemma);
        
        // Section 5 & 6: POS
        if (posTagging.length > 0) renderTable('pos-table', posTagging);
        
        // Section 7: NER
        if (ner.length > 0) renderTable('ner-table', ner);
        
        // Section 8: N-Gram
        if (ngramStats.length > 0) {
            renderTable('ngram-table', ngramStats);
            
            // Extract top 5-grams
            const fiveGramRow = ngramStats.find(r => r['N-Gram'] === '5-Gram');
            if (fiveGramRow && fiveGramRow['Top N-Grams']) {
                const topStr = fiveGramRow['Top N-Grams'];
                // Format: "shall (116); employee (85); ..."
                const regex = /(.+?)\s*\((\d+)\)(?:;\s*|$)/g;
                let match;
                let tagsHtml = '';
                while ((match = regex.exec(topStr)) !== null) {
                    tagsHtml += `<div class="tag">${match[1]} <span class="tag-count">${match[2]}</span></div>`;
                }
                document.getElementById('top-ngrams-container').innerHTML = tagsHtml;
            }
        }
        
        // Section 9: BPE
        if (bpe.length > 0) renderTable('bpe-table', bpe);
        
        // Section 11 & 13: Retrieval (Stored Queries)
        if (retrieval.length > 0) {
            state.retrieval = retrieval;
            const selectEl = document.getElementById('stored-queries');
            
            // Group by query text since each has 3 pipelines. We only want unique query strings.
            const uniqueQueries = [...new Set(retrieval.map(r => r.Query))];
            uniqueQueries.forEach(q => {
                const opt = document.createElement('option');
                opt.value = q;
                opt.textContent = q;
                selectEl.appendChild(opt);
            });
            
            selectEl.addEventListener('change', (e) => {
                const qStr = e.target.value;
                if (!qStr) {
                    document.getElementById('results-panel').classList.add('hidden');
                    document.getElementById('query-details').classList.add('hidden');
                    return;
                }
                
                // For simplicity, default to "Final Pipeline" results for that query
                const resultRow = retrieval.find(r => r.Query === qStr && r.Pipeline === 'Final Pipeline') || retrieval.find(r => r.Query === qStr);
                
                if (resultRow) {
                    document.getElementById('query-details').classList.remove('hidden');
                    document.getElementById('q-type').textContent = resultRow.Type || 'Unknown';
                    document.getElementById('q-pipeline').textContent = resultRow.Pipeline || 'Unknown';
                    document.getElementById('q-time').textContent = resultRow['Time (ms)'] || '0';
                    
                    const numResults = resultRow['# Results'] || '0';
                    document.getElementById('results-count').textContent = numResults;
                    
                    const docsStr = resultRow['Retrieved Documents'];
                    const docsContainer = document.getElementById('retrieval-results-container');
                    
                    if (!docsStr || docsStr.trim() === '' || docsStr === '[]') {
                        docsContainer.innerHTML = '<p class="text-muted">No documents retrieved.</p>';
                    } else {
                        const docsList = docsStr.split(' ').filter(d => d.trim().length > 0);
                        let html = '';
                        docsList.forEach((dId, idx) => {
                            html += `
                            <div class="result-item">
                                <span class="result-id">${dId}</span>
                                <span class="result-rank">Rank ${idx + 1}</span>
                            </div>`;
                        });
                        docsContainer.innerHTML = html;
                    }
                    document.getElementById('results-panel').classList.remove('hidden');
                }
            });
        }
        
        // Section 14 & 15: Evaluation
        if (pipelineComp.length > 0) renderTable('pipeline-comp-table', pipelineComp);
        if (evaluation.length > 0) renderTable('eval-table', evaluation);

    } catch (e) {
        console.error("Initialization Error:", e);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error initializing app: ${e.message}</p>`;
    }
}

// Upload UI Logic
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const processBtn = document.getElementById('process-btn');
let selectedFiles = [];

function updateFileList() {
    fileList.innerHTML = '';
    if (selectedFiles.length > 0) {
        processBtn.classList.remove('hidden');
        selectedFiles.forEach((file, index) => {
            const size = (file.size / 1024).toFixed(1) + ' KB';
            fileList.innerHTML += `
                <div class="file-item">
                    <span>📄 <strong>${file.name}</strong> (${size})</span>
                    <span style="color:var(--text-muted); cursor:pointer;" onclick="removeFile(${index})">❌</span>
                </div>
            `;
        });
    } else {
        processBtn.classList.add('hidden');
    }
}

window.removeFile = function(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
};

if (dropZone && fileInput) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            selectedFiles = [...selectedFiles, ...Array.from(e.dataTransfer.files)];
            updateFileList();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            selectedFiles = [...selectedFiles, ...Array.from(e.target.files)];
            updateFileList();
        }
    });

    processBtn.addEventListener('click', () => {
        alert("In a fully connected backend (Tier 2), this would upload your " + selectedFiles.length + " document(s) to the Python NLP pipeline for extraction, OCR, and analysis.");
    });
}

// Live Search Logic (Tier 2)
const searchBtn = document.getElementById('search-btn');
const queryInput = document.getElementById('query-input');
const queryType = document.getElementById('query-type');

if (searchBtn && queryInput) {
    searchBtn.addEventListener('click', async () => {
        let qStr = queryInput.value.trim();
        if (!qStr) return;
        
        // Ensure formatting for boolean if dropdown is used
        const type = queryType.value;
        if (type.startsWith("Boolean")) {
            const operator = type.split(" ")[1];
            // Basic heuristic if they typed two words without the operator
            if (!qStr.includes(" AND ") && !qStr.includes(" OR ") && !qStr.includes(" NOT ")) {
                const words = qStr.split(" ");
                if (words.length >= 2) {
                    qStr = words.join(` ${operator} `);
                }
            }
        } else if (type === "Phrase" && !qStr.startsWith('"')) {
            qStr = `"${qStr}"`;
        }
        
        try {
            searchBtn.textContent = 'Searching...';
            // Call Flask API on port 5000
            const response = await fetch(`http://localhost:5000/api/search?q=${encodeURIComponent(qStr)}`);
            const data = await response.json();
            searchBtn.textContent = 'Search';
            
            if (data.error) {
                alert("Backend Error: " + data.error);
                return;
            }
            
            document.getElementById('query-details').classList.remove('hidden');
            document.getElementById('q-type').textContent = data.type || type;
            document.getElementById('q-pipeline').textContent = data.pipeline || 'Final Pipeline';
            document.getElementById('q-time').textContent = data.time_ms || '0';
            
            const docsList = data.results || [];
            document.getElementById('results-count').textContent = docsList.length;
            
            const docsContainer = document.getElementById('retrieval-results-container');
            if (docsList.length === 0) {
                docsContainer.innerHTML = '<p class="text-muted">No documents retrieved.</p>';
            } else {
                let html = '';
                docsList.forEach((dId, idx) => {
                    html += `
                    <div class="result-item">
                        <span class="result-id">${dId}</span>
                        <span class="result-rank">Rank ${idx + 1}</span>
                    </div>`;
                });
                docsContainer.innerHTML = html;
            }
            document.getElementById('results-panel').classList.remove('hidden');
            
        } catch (e) {
            searchBtn.textContent = 'Search';
            alert("Could not connect to the Python backend! Please make sure you are running 'python server.py' in your terminal.");
        }
    });
    
    // Allow 'Enter' key to trigger search
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchBtn.click();
    });
}

// Pipeline Analysis Logic (Tier 2)
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeInput = document.getElementById('analyze-input');

if (analyzeBtn && analyzeInput) {
    analyzeBtn.addEventListener('click', async () => {
        const text = analyzeInput.value.trim();
        if (!text) return;
        
        try {
            analyzeBtn.textContent = 'Analyzing...';
            const response = await fetch(`http://localhost:5000/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            analyzeBtn.textContent = 'Analyze Text';
            
            if (data.error) {
                alert("Backend Error: " + data.error);
                return;
            }
            
            document.getElementById('analyze-results').classList.remove('hidden');
            document.getElementById('analyze-raw').textContent = JSON.stringify(data.raw, null, 2);
            document.getElementById('analyze-final').textContent = JSON.stringify(data.final, null, 2);
            document.getElementById('analyze-pos').textContent = JSON.stringify(data.pos, null, 2);
            
        } catch (e) {
            analyzeBtn.textContent = 'Analyze Text';
            alert("Could not connect to the Python backend! Please make sure you are running 'python server.py' in your terminal.");
        }
    });
}

// Start app
window.addEventListener('DOMContentLoaded', init);
