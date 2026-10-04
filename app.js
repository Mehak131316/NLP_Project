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

        // Populate Corpus PDF selector
        loadCorpusFilesList();

    } catch (e) {
        console.error("Initialization Error:", e);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error initializing app: ${e.message}</p>`;
    }
}

// -------------------------------------------------------------
// Document Processing & Stage Timing UI Logic
// -------------------------------------------------------------
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const processBtn = document.getElementById('process-btn');
const corpusSelect = document.getElementById('corpus-file-select');
const processCorpusBtn = document.getElementById('process-corpus-btn');
let selectedFiles = [];

// Load Available Corpus Files into Select
async function loadCorpusFilesList() {
    try {
        const res = await fetch('http://localhost:5000/api/available-docs');
        const data = await res.json();
        if (data.files && data.files.length > 0) {
            corpusSelect.innerHTML = '<option value="">-- Choose a Corpus PDF / Contract (32 Files) --</option>';
            data.files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f.name;
                opt.textContent = `${f.name} (${f.size_kb} KB)`;
                corpusSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn("Could not fetch corpus files:", e);
    }
}
loadCorpusFilesList();

if (corpusSelect && processCorpusBtn) {
    corpusSelect.addEventListener('change', (e) => {
        processCorpusBtn.disabled = !e.target.value;
    });

    processCorpusBtn.addEventListener('click', async () => {
        const filename = corpusSelect.value;
        if (!filename) return;
        await executeDocumentProcessing({ filename: filename }, filename);
    });
}

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
            selectedFiles = [e.dataTransfer.files[0]]; // Process 1 at a time for deep analysis
            updateFileList();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            selectedFiles = [e.target.files[0]];
            updateFileList();
        }
    });

    processBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;
        const file = selectedFiles[0];
        const formData = new FormData();
        formData.append('file', file);
        await executeDocumentProcessing(formData, file.name);
    });
}

// Execute Document Pipeline on Backend with Animated Progress
async function executeDocumentProcessing(payload, displayName) {
    const loader = document.getElementById('processing-loader');
    const resultPanel = document.getElementById('doc-result-panel');
    const statusTitle = document.getElementById('processing-status-title');
    const statusStep = document.getElementById('processing-status-step');

    loader.classList.remove('hidden');
    resultPanel.classList.add('hidden');
    statusTitle.textContent = `Processing ${displayName}...`;

    const steps = [
        "Step 1/8: Extracting text & running RapidOCR for scanned pages...",
        "Step 2/8: Conservative cleaning & de-hyphenating wrapped lines...",
        "Step 3/8: Normalizing parenthetical numbers & durations...",
        "Step 4/8: Running Built-in, Custom & Hybrid Tokenizers...",
        "Step 5/8: Running POS Tagging (spaCy & Custom Rules)...",
        "Step 6/8: Extracting Named Entities (MONEY, LAW, DURATION, DESIGNATION)...",
        "Step 7/8: Generating sentence-bounded N-Grams & Legal Obligations...",
        "Step 8/8: Computing Pipeline A, Pipeline B & Final Pipeline transforms..."
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
        if (stepIdx < steps.length) {
            statusStep.textContent = steps[stepIdx];
            stepIdx++;
        }
    }, 150);

    try {
        let response;
        if (payload instanceof FormData) {
            response = await fetch('http://localhost:5000/api/process-doc', {
                method: 'POST',
                body: payload
            });
        } else {
            response = await fetch('http://localhost:5000/api/process-doc', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        }

        clearInterval(interval);
        const data = await response.json();
        loader.classList.add('hidden');

        if (data.error) {
            alert("Processing Error: " + data.error);
            return;
        }

        renderDocumentProcessingResults(data);
        resultPanel.classList.remove('hidden');
        resultPanel.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(interval);
        loader.classList.add('hidden');
        alert("Failed to connect to Python backend! Please make sure 'python server.py' is running on port 5000.");
        console.error(err);
    }
}

// Render Document Results & Timing
function renderDocumentProcessingResults(data) {
    // 1. Hero Banner
    document.getElementById('res-filename').textContent = data.filename;
    document.getElementById('res-summary-sub').textContent = 
        `Format: ${data.format} | Total Pages: ${data.pages} | OCR Enabled: ${data.ocr_used ? '✓ Yes (' + data.ocr_pages + ' pages OCR\'d)' : 'No (Native Text Layer)'}`;
    document.getElementById('res-total-time').textContent = `${data.total_time_ms.toFixed(2)} ms`;
    document.getElementById('res-time-sec').textContent = `(${data.total_time_sec.toFixed(3)} seconds)`;

    // Timing Pills
    const timings = data.timings || {};
    const pillDefs = [
        { label: "Extraction / OCR", key: "extraction_ms" },
        { label: "Cleaning & Normalization", key: "cleaning_ms" },
        { label: "Tokenization", key: "tokenization_ms" },
        { label: "POS Tagging", key: "pos_tagging_ms" },
        { label: "NER Analysis", key: "ner_ms" },
        { label: "Lemmatization & Stopwords", key: "lemmatization_stopwords_ms" },
        { label: "Legal Obligations", key: "obligations_ms" },
        { label: "N-Grams", key: "ngrams_ms" },
        { label: "Pipeline Transforms", key: "pipelines_ms" },
    ];

    let pillsHtml = '';
    pillDefs.forEach(p => {
        const val = timings[p.key] !== undefined ? timings[p.key] : 0;
        pillsHtml += `<div class="timing-pill">${p.label}: <strong>${val.toFixed(2)} ms</strong></div>`;
    });
    document.getElementById('res-timing-pills').innerHTML = pillsHtml;

    // 2. Summary Metric Cards
    const stats = data.statistics || {};
    document.getElementById('doc-metric-cards').innerHTML = `
        <div class="stat-card">
            <div class="label">Total Characters</div>
            <div class="value">${(stats.characters || 0).toLocaleString()}</div>
        </div>
        <div class="stat-card">
            <div class="label">Sentences Detected</div>
            <div class="value">${stats.sentences_count || 0}</div>
        </div>
        <div class="stat-card">
            <div class="label">Raw Tokens</div>
            <div class="value">${(stats.raw_tokens_count || 0).toLocaleString()}</div>
        </div>
        <div class="stat-card">
            <div class="label">Final Pipeline Tokens</div>
            <div class="value" style="color:var(--accent-green);">${(stats.final_tokens_count || 0).toLocaleString()}</div>
        </div>
        <div class="stat-card">
            <div class="label">Unique Terms</div>
            <div class="value">${(stats.vocabulary_size || 0).toLocaleString()}</div>
        </div>
    `;

    // 3. Tab 1: NER Badges
    const entityColors = {
        "MONEY": { bg: "#fce7f3", border: "#ec4899", text: "#9d174d" },
        "LAW": { bg: "#fee2e2", border: "#ef4444", text: "#991b1b" },
        "DURATION": { bg: "#cffafe", border: "#06b6d4", text: "#155e75" },
        "DATE": { bg: "#fef3c7", border: "#f59e0b", text: "#92400e" },
        "DESIGNATION": { bg: "#e0e7ff", border: "#6366f1", text: "#3730a3" },
        "ORG": { bg: "#ede9fe", border: "#8b5cf6", text: "#5b21b6" },
        "PERSON": { bg: "#dbeafe", border: "#3b82f6", text: "#1e40af" },
        "GPE": { bg: "#dcfce7", border: "#10b981", text: "#166534" },
    };

    let badgesHtml = '';
    if (data.entities && data.entities.length > 0) {
        data.entities.forEach(ent => {
            const style = entityColors[ent.label] || { bg: "#f1f5f9", border: "#94a3b8", text: "#334155" };
            badgesHtml += `
                <span class="entity-badge" style="background-color:${style.bg}; border-color:${style.border}; color:${style.text};">
                    <strong>${ent.text}</strong>
                    <span style="background-color:${style.border}; color:#ffffff; font-size:10px; padding:1px 5px; border-radius:3px; margin-left:4px; text-transform:uppercase; font-weight:bold;">${ent.label}</span>
                </span>
            `;
        });
    } else {
        badgesHtml = '<p class="text-muted">No named entities detected in document.</p>';
    }
    document.getElementById('res-entities-badges').innerHTML = badgesHtml;

    // 4. Tab 2: POS Tags Table
    const posList = data.pos_tags || [];
    renderTable('res-pos-table', posList.slice(0, 100).map(p => ({
        "Word": p.word,
        "spaCy Penn Tag": p.spacy_tag,
        "Custom Domain Rule Tag": p.rule_tag
    })));

    // 5. Tab 3: Obligations Table
    const oblList = data.obligations || [];
    renderTable('res-obligations-table', oblList.map(o => ({
        "Subject / Party": o.subject || '(implied)',
        "Modal": o.modal,
        "Negated?": o.negated ? '✓ YES (Prohibition)' : 'No',
        "Verb Action": o.verb,
        "Object / Scope": o.object || '-',
        "Classification": o.type
    })));

    // 6. Tab 4: Pipeline Transforms
    const pipeComp = data.pipeline_comparison || {};
    document.getElementById('res-pipe-a-tokens').textContent = (pipeComp['Pipeline A']?.sample || []).join(', ') + '...';
    document.getElementById('res-pipe-b-tokens').textContent = (pipeComp['Pipeline B']?.sample || []).join(', ') + '...';
    document.getElementById('res-pipe-final-tokens').textContent = (pipeComp['Final Pipeline']?.sample || []).join(', ') + '...';

    // 7. Tab 5: N-Grams Tables
    renderTable('res-bigrams-table', data.ngrams?.bigrams || []);
    renderTable('res-trigrams-table', data.ngrams?.trigrams || []);

    // 8. Tab 6: Extracted Text
    let showingRaw = false;
    const textPre = document.getElementById('res-text-content');
    const toggleBtn = document.getElementById('toggle-raw-text-btn');
    textPre.textContent = data.text_preview;

    toggleBtn.onclick = () => {
        showingRaw = !showingRaw;
        if (showingRaw) {
            textPre.textContent = data.raw_text_preview || data.text_preview;
            toggleBtn.textContent = "Show Cleaned Normalized Text";
        } else {
            textPre.textContent = data.text_preview;
            toggleBtn.textContent = "Show Raw Uncleaned Text";
        }
    };
}

// Sub-Tab Switching Logic inside Document Results Panel
document.querySelectorAll('.doc-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.doc-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.doc-tab-content').forEach(c => {
            c.classList.remove('active');
            c.classList.add('hidden');
        });

        const targetBtn = e.currentTarget;
        targetBtn.classList.add('active');
        const targetTabId = targetBtn.getAttribute('data-tab');
        const targetContent = document.getElementById(targetTabId);
        if (targetContent) {
            targetContent.classList.remove('hidden');
            targetContent.classList.add('active');
        }
    });
});


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
