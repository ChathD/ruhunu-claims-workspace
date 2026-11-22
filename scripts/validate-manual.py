"""Exercise one synthetic manual PDF submission through the real Claude workflow."""
import base64
import io
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from sqlalchemy import select
from backend.api import create_app
from backend.storage.sqlite_store import Assessment, ModelInvocation
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.model_settings import settings
from backend.evidence.policy_wording_index import create_document, SYNTHETIC_TEXT

if not settings()[0]:
    raise SystemExit("Enable Claude before running the live manual smoke test")

writer = PdfWriter()
page = writer.add_blank_page(width=600, height=800)
font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
content = DecodedStreamObject()
lines = ["SYNTHETIC TEST ESTIMATE EST-MANUAL-SMOKE", "Registration: ABC-SMOKE", "Estimate date: 2026-10-16",
         "Garage: Sample Auto Glass", "Garage address: 25 Sample Road, Colombo",
         "Windscreen replacement: LKR 100000.00", "Total: LKR 100000.00"]
content.set_data(('BT /F1 12 Tf 30 750 Td ' + ' 0 -22 Td '.join('(' + line + ') Tj' for line in lines) + ' ET').encode())
page[NameObject('/Contents')] = writer._add_object(content)
stream = io.BytesIO()
writer.write(stream)

directory = ROOT / '.local' / 'manual-validation' / str(uuid.uuid4())
app = create_app(directory, auto_seed=False)
create_document(app.state.store, title='Manual smoke-test motor wording', product='SMOKE-TEST', version='1', filename='motor.txt',
                media_type='text/plain', content=base64.b64encode(SYNTHETIC_TEXT.encode()).decode())
with TestClient(app) as client:
    client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
    response = client.post('/api/portal/claims', json={'policy_id': 'MANUAL-SMOKE-POL', 'registration': 'ABC-SMOKE',
        'loss_type': 'glass', 'loss_date': '2026-10-15', 'gross_amount': '100000.00',
        'narrative': 'A stone cracked the windscreen. Synthetic manual workflow test.', 'attachments': [
            {'name': 'estimate.pdf', 'media_type': 'application/pdf', 'content': base64.b64encode(stream.getvalue()).decode()}]})
    response.raise_for_status()
    identifier = response.json()['id']
    print('Queued manual assessment:', identifier, flush=True)
    previous = None
    for _ in range(600):
        with app.state.store.session() as session:
            assessment = session.get(Assessment, identifier)
            calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier).order_by(ModelInvocation.created_at)).all()
        progress = (assessment.status, len(calls), calls[-1].stage if calls else 'waiting')
        if progress != previous:
            print('Progress:', progress, flush=True)
            previous = progress
        if assessment.status in {'complete', 'failed'}:
            break
        time.sleep(1)
    if assessment.status != 'complete':
        raise SystemExit('Manual workflow did not complete')
    package = export_audit(app.state.store, identifier)
    (directory / 'audit.json').write_text(json.dumps(package, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(directory), 'route': package['decision']['route'],
        'specialists': [run['stage'] for run in package['agent_runs']],
        'extraction': [{'name': item['name'], 'status': item['status']} for item in assessment.context.get('attachment_extraction', [])],
        'tools': sorted({item['tool'] for item in package['retrievals']}), 'model_calls': len(calls),
        'failed_calls': sum(call.status == 'failed' for call in calls), 'audit_valid': verify_audit(package)}, indent=2), flush=True)
