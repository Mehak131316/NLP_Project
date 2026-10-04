import os
import sys
import json
from document_processor import process_single_document

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sample_pdf = os.path.join('NLP_Assessment1-20261003T034029Z-1-001', 'NLP_Assessment1', 'data', 'EMPLOYMENT_AGREEMENT Startup.pdf')
if not os.path.exists(sample_pdf):
    data_dir = os.path.join('NLP_Assessment1-20261003T034029Z-1-001', 'NLP_Assessment1', 'data')
    files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    sample_pdf = os.path.join(data_dir, files[0])

print(f"Testing on PDF: {sample_pdf}")
res = process_single_document(sample_pdf, os.path.basename(sample_pdf))
print(f"Success: {res['success']}")
print(f"Total Time: {res['total_time_ms']} ms ({res['total_time_sec']} s)")
print("Timings breakdown:")
print(json.dumps(res['timings'], indent=2))
print("Statistics:")
print(json.dumps(res['statistics'], indent=2))
print(f"Entities found: {len(res['entities'])}")
print(f"Obligations found: {len(res['obligations'])}")
