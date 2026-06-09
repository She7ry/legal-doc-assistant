from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path
from textwrap import wrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import file_sha256, load_documents
from doc_assistant.retrieval.vector_store import _chunk_text_with_heading, _split_legal_sections

FIXTURE_DIR = PROJECT_ROOT / "data" / "eval" / "fixtures"
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"


@dataclass(frozen=True)
class EvalDocument:
    file_name: str
    pages: list[list[str]]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    answer_type: str
    gold_answer: str
    markers: tuple[str, ...] = ()
    required_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    required_refusal_terms: tuple[str, ...] = field(default_factory=tuple)


DOCUMENTS = [
    EvalDocument(
        file_name="sample_supply_contract.pdf",
        pages=[
            [
                "ACME SUPPLY AGREEMENT",
                "Document ID: EVAL-CONTRACT-001",
                "",
                "Section 1. Parties",
                "Buyer: Northwind Retail LLC. Supplier: Blue Harbor Components Ltd.",
                "",
                "Section 2. Delivery",
                "Supplier must deliver conforming goods to Buyer's Shanghai warehouse on or before the delivery date stated in each purchase order.",
                "A delivery is late if the goods arrive more than five business days after the applicable delivery date.",
                "",
                "Section 5. Payment",
                "Marker: EVAL-C-5.1.",
                "Buyer must pay undisputed invoices within 30 calendar days after receiving a valid invoice.",
                "Disputed invoice amounts may be withheld until the dispute is resolved in writing by both parties.",
                "",
                "Section 8.1 Termination for Convenience",
                "Marker: EVAL-C-8.1.",
                "Either party may terminate this agreement for convenience by giving at least 30 days' prior written notice to the other party.",
                "Termination for convenience does not waive payment obligations that accrued before the effective termination date.",
            ],
            [
                "ACME SUPPLY AGREEMENT",
                "Document ID: EVAL-CONTRACT-001",
                "",
                "Section 9. Warranty",
                "Marker: EVAL-C-9.1.",
                "Supplier warrants that delivered goods will conform to the specifications for 12 months after acceptance.",
                "Buyer's exclusive remedy for breach of this product warranty is repair, replacement, or refund of the affected goods.",
                "",
                "Section 12.2 Liquidated Damages",
                "Marker: EVAL-C-12.2.",
                "If Supplier delivers goods more than five business days late, Supplier must pay liquidated damages equal to 0.5% of the delayed shipment value for each day of delay.",
                "Liquidated damages are capped at 10% of the delayed shipment value.",
                "Payment of liquidated damages does not limit Buyer's right to reject nonconforming goods.",
                "",
                "Section 14. Confidentiality",
                "Marker: EVAL-C-14.1.",
                "Each party must protect the other party's confidential information using reasonable care and may use it only to perform this agreement.",
                "The confidentiality obligation survives for three years after termination.",
            ],
            [
                "ACME SUPPLY AGREEMENT",
                "Document ID: EVAL-CONTRACT-001",
                "",
                "Section 16. Records and Audit",
                "Marker: EVAL-C-16.1.",
                "Supplier must maintain complete delivery, testing, and invoice records for two years after final payment.",
                "Buyer may audit those records once per calendar year on 10 business days' notice.",
                "",
                "Section 18. Governing Law",
                "This agreement is governed by the laws of the State of New York, excluding conflict of laws rules.",
                "The parties consent to exclusive venue in the state or federal courts located in New York County, New York.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_saas_msa.pdf",
        pages=[
            [
                "ORION ANALYTICS MASTER SERVICES AGREEMENT",
                "Document ID: EVAL-MSA-002",
                "",
                "Section 1. Services",
                "Provider will make the hosted analytics platform available to Customer and its authorized users during the subscription term.",
                "Customer is responsible for user account administration and for the accuracy of data submitted to the service.",
                "",
                "Section 3.2 Fees and Late Payment",
                "Marker: EVAL-MSA-3.2.",
                "Subscription fees are invoiced annually in advance unless an order form states a different billing schedule.",
                "Overdue undisputed amounts accrue interest at 1.5% per month or the maximum rate permitted by law, whichever is lower.",
                "",
                "Section 4.1 Service Level",
                "Marker: EVAL-MSA-4.1.",
                "Provider will use commercially reasonable efforts to make the production service available 99.9% of each calendar month.",
                "If availability falls below 99.9%, Customer's sole remedy is a service credit equal to 5% of the affected monthly subscription fee.",
            ],
            [
                "ORION ANALYTICS MASTER SERVICES AGREEMENT",
                "Document ID: EVAL-MSA-002",
                "",
                "Section 7.3 Limitation of Liability",
                "Marker: EVAL-MSA-7.3.",
                "Except for excluded claims, each party's aggregate liability is capped at the fees paid or payable in the 12 months before the event giving rise to the claim.",
                "For claims arising from unauthorized disclosure of Customer Data, Provider's aggregate liability cap is two times the fees paid or payable in that same 12 month period.",
                "",
                "Section 9.2 Termination for Cause",
                "Marker: EVAL-MSA-9.2.",
                "Either party may terminate this agreement for material breach if the breach remains uncured 60 days after written notice.",
                "Provider may suspend access sooner if Customer's use creates a security risk or violates applicable law.",
                "",
                "Section 11.4 Data Export",
                "Marker: EVAL-MSA-11.4.",
                "For 30 days after termination, Provider will make Customer Data available for export in CSV or JSON format.",
                "After the export period, Provider may delete Customer Data from active systems according to its standard retention schedule.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_data_processing_addendum.pdf",
        pages=[
            [
                "DATA PROCESSING ADDENDUM",
                "Document ID: EVAL-DPA-003",
                "",
                "Section 2.1 Processor Instructions",
                "Marker: EVAL-DPA-2.1.",
                "Processor will process Personal Data only on documented instructions from Controller, including instructions in the agreement and applicable order forms.",
                "Processor will promptly notify Controller if it believes an instruction violates applicable data protection law.",
                "",
                "Section 4.2 Technical and Organizational Measures",
                "Marker: EVAL-DPA-4.2.",
                "Processor must encrypt Personal Data at rest using AES-256 or stronger encryption and encrypt Personal Data in transit using TLS 1.2 or higher.",
                "Processor must maintain access controls, logging, vulnerability management, and annual workforce privacy training.",
            ],
            [
                "DATA PROCESSING ADDENDUM",
                "Document ID: EVAL-DPA-003",
                "",
                "Section 5.1 Subprocessors",
                "Marker: EVAL-DPA-5.1.",
                "Processor may appoint subprocessors only after giving Controller at least 15 days' prior notice through the designated subprocessor notice portal.",
                "Controller may object on reasonable data protection grounds during the notice period.",
                "",
                "Section 6.3 Security Incident Notice",
                "Marker: EVAL-DPA-6.3.",
                "Processor must notify Controller without undue delay and no later than 48 hours after confirming a Security Incident involving Personal Data.",
                "The notice must describe the nature of the incident, affected data categories, mitigation steps, and contact information.",
                "",
                "Section 8.1 Return or Deletion",
                "Marker: EVAL-DPA-8.1.",
                "Upon termination, Processor must return or delete Personal Data within 45 days unless applicable law requires retention.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_procurement_policy.pdf",
        pages=[
            [
                "GLOBAL PROCUREMENT POLICY",
                "Document ID: EVAL-PROC-004",
                "",
                "Section 2. Scope",
                "This policy applies to purchases of goods, software, services, contractors, and renewals made by company personnel.",
                "Emergency purchases may proceed before formal approval only when delay would materially harm safety, legal compliance, or service continuity.",
                "",
                "Section 4.1 Competitive Quotes",
                "Marker: EVAL-PROC-4.1.",
                "Purchases over USD 25,000 require at least three written quotes unless Legal and Procurement approve a documented sole-source justification.",
                "Purchases over USD 100,000 also require CFO approval before a purchase order may be issued.",
                "",
                "Section 4.4 Anti-Splitting Rule",
                "Marker: EVAL-PROC-4.4.",
                "Employees may not split related purchases into smaller orders to avoid approval thresholds or competitive quote requirements.",
            ],
            [
                "GLOBAL PROCUREMENT POLICY",
                "Document ID: EVAL-PROC-004",
                "",
                "Section 6.1 Payment Terms",
                "Marker: EVAL-PROC-6.1.",
                "Standard supplier payment terms are net 45 days from receipt of a valid invoice unless Finance approves a shorter term.",
                "Prepayment requires written approval from the budget owner and Finance.",
                "",
                "Section 7.2 SaaS Vendor Risk Review",
                "Marker: EVAL-PROC-7.2.",
                "SaaS vendors that store Confidential Information must complete security review and provide a current SOC 2 Type II report or equivalent independent assessment.",
                "High risk vendors must have a remediation plan accepted by Security before contract signature.",
                "",
                "Section 9. Records",
                "Approved purchase requests, quotes, sole-source justifications, and signed contracts must be retained for seven years.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_information_security_policy.pdf",
        pages=[
            [
                "INFORMATION SECURITY POLICY",
                "Document ID: EVAL-SEC-005",
                "",
                "Section 3.1 Multi-Factor Authentication",
                "Marker: EVAL-SEC-3.1.",
                "Multi-factor authentication is required for privileged accounts, remote access, production systems, and any application storing Confidential Information.",
                "Shared accounts are prohibited unless Security grants a time-limited exception.",
                "",
                "Section 4.2 Encryption",
                "Marker: EVAL-SEC-4.2.",
                "Confidential Information must be encrypted at rest with AES-256 or an approved equivalent and encrypted in transit with TLS 1.2 or higher.",
                "Encryption keys must be rotated at least annually and whenever compromise is suspected.",
            ],
            [
                "INFORMATION SECURITY POLICY",
                "Document ID: EVAL-SEC-005",
                "",
                "Section 5.2 Vulnerability Remediation",
                "Marker: EVAL-SEC-5.2.",
                "Critical vulnerabilities on internet-facing systems must be remediated or mitigated within seven calendar days after validation.",
                "High vulnerabilities must be remediated or mitigated within 30 calendar days after validation.",
                "",
                "Section 6.4 Logging",
                "Marker: EVAL-SEC-6.4.",
                "Production authentication logs, administrative activity logs, and security event logs must be retained for at least one year.",
                "Security may require longer retention for regulated systems or active investigations.",
                "",
                "Section 8.3 Vendor Incident Notice",
                "Marker: EVAL-SEC-8.3.",
                "Vendors handling Confidential Information must notify Security within 24 hours after discovering a confirmed or suspected security incident.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_employee_handbook.pdf",
        pages=[
            [
                "EMPLOYEE HANDBOOK EXCERPT",
                "Document ID: EVAL-HR-006",
                "",
                "Section 2.1 Remote Work",
                "Marker: EVAL-HR-2.1.",
                "Eligible employees may work remotely up to three days per week with manager approval and must remain reachable during core collaboration hours.",
                "Remote work approval may be modified based on role requirements, performance, or business needs.",
                "",
                "Section 4.3 Paid Time Off",
                "Marker: EVAL-HR-4.3.",
                "Full-time employees accrue 15 days of paid time off per calendar year, prorated for new hires and part-time schedules.",
                "Unused paid time off does not carry over unless local law requires otherwise.",
            ],
            [
                "EMPLOYEE HANDBOOK EXCERPT",
                "Document ID: EVAL-HR-006",
                "",
                "Section 5.2 Overtime Approval",
                "Marker: EVAL-HR-5.2.",
                "Non-exempt employees must obtain manager approval before working overtime, except in emergencies where prior approval is impracticable.",
                "All hours worked must be recorded accurately even if overtime was not pre-approved.",
                "",
                "Section 6.1 Expense Reimbursement",
                "Marker: EVAL-HR-6.1.",
                "Employees must submit reimbursement requests with receipts within 30 days after incurring the business expense.",
                "The company reimburses approved expenses through payroll or accounts payable according to the regular payment cycle.",
                "",
                "Section 9. Confidentiality",
                "Employees must protect company confidential information during and after employment.",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_mutual_nda.pdf",
        pages=[
            [
                "MUTUAL NON-DISCLOSURE AGREEMENT",
                "Document ID: EVAL-NDA-007",
                "",
                "Section 2. Definition of Confidential Information",
                "Confidential Information includes non-public technical, business, financial, customer, product, and security information disclosed by either party.",
                "Information is not confidential if it is publicly available without breach, already known without restriction, independently developed, or rightfully received from a third party.",
                "",
                "Section 4.1 Term of Confidentiality",
                "Marker: EVAL-NDA-4.1.",
                "Each party's confidentiality obligations continue for five years after disclosure.",
                "Trade secrets remain protected for as long as they qualify as trade secrets under applicable law.",
            ],
            [
                "MUTUAL NON-DISCLOSURE AGREEMENT",
                "Document ID: EVAL-NDA-007",
                "",
                "Section 6.2 Return or Destruction",
                "Marker: EVAL-NDA-6.2.",
                "Upon written request, the receiving party must return or destroy Confidential Information within 10 business days.",
                "One archival copy may be retained solely for legal compliance and backup purposes, subject to continuing confidentiality obligations.",
                "",
                "Section 8.1 Equitable Relief",
                "Marker: EVAL-NDA-8.1.",
                "The parties agree that unauthorized disclosure may cause irreparable harm for which monetary damages are inadequate.",
                "The disclosing party may seek injunctive or other equitable relief without posting bond where permitted by law.",
            ],
        ],
    ),
]


REFUSAL_TERMS = (
    "not found",
    "not provided",
    "cannot determine",
    "not enough information",
    "relevant text was not found",
    "did not find enough relevant text",
    "do not contain",
    "does not contain",
    "do not specify",
    "does not specify",
    "do not mention",
    "does not mention",
)


CASES = [
    EvalCase(
        id="eval_001_liquidated_damages_cap",
        question="What is the cap on liquidated damages for late delivery?",
        answer_type="answerable",
        gold_answer="Liquidated damages are capped at 10% of the delayed shipment value.",
        markers=("EVAL-C-12.2",),
        required_answer_terms=("10%", "delayed shipment value"),
        forbidden_answer_terms=("20%", "contract total value"),
    ),
    EvalCase(
        id="eval_002_cyber_insurance_refusal",
        question="Does the agreement require Supplier to buy cybersecurity insurance?",
        answer_type="unanswerable",
        gold_answer="The indexed documents do not state a cybersecurity insurance requirement.",
    ),
    EvalCase(
        id="eval_003_notice_period",
        question="How much notice is required before termination for convenience?",
        answer_type="answerable",
        gold_answer="Either party may terminate for convenience with 30 days' prior written notice.",
        markers=("EVAL-C-8.1",),
        required_answer_terms=("30 days", "written notice"),
        forbidden_answer_terms=("60 days", "90 days"),
    ),
    EvalCase(
        id="eval_004_governing_law_refusal",
        question="Which state's employment law governs wrongful termination claims for employees?",
        answer_type="unanswerable",
        gold_answer="The indexed supply contract does not address employment wrongful termination law.",
    ),
    EvalCase(
        id="eval_005_invoice_payment_period",
        question="When must Buyer pay undisputed invoices under the supply agreement?",
        answer_type="answerable",
        gold_answer="Buyer must pay undisputed invoices within 30 calendar days after receiving a valid invoice.",
        markers=("EVAL-C-5.1",),
        required_answer_terms=("30 calendar days", "valid invoice"),
        forbidden_answer_terms=("45 days", "annual"),
    ),
    EvalCase(
        id="eval_006_product_warranty_duration",
        question="How long does the product warranty last after acceptance?",
        answer_type="answerable",
        gold_answer="The delivered goods warranty lasts for 12 months after acceptance.",
        markers=("EVAL-C-9.1",),
        required_answer_terms=("12 months", "acceptance"),
        forbidden_answer_terms=("five years", "30 days"),
    ),
    EvalCase(
        id="eval_007_supplier_record_retention",
        question="How long must Supplier maintain delivery and invoice records?",
        answer_type="answerable",
        gold_answer="Supplier must maintain delivery, testing, and invoice records for two years after final payment.",
        markers=("EVAL-C-16.1",),
        required_answer_terms=("two years", "final payment"),
        forbidden_answer_terms=("seven years", "one year"),
    ),
    EvalCase(
        id="eval_008_supplier_delay_cap_chinese",
        question="供应商延迟交付的违约金上限是多少?",
        answer_type="answerable",
        gold_answer="The cap is 10% of the delayed shipment value.",
        markers=("EVAL-C-12.2",),
        required_answer_terms=("10%",),
        forbidden_answer_terms=("20%",),
    ),
    EvalCase(
        id="eval_009_saas_late_payment_interest",
        question="What interest rate applies to overdue undisputed SaaS fees?",
        answer_type="answerable",
        gold_answer="Overdue undisputed amounts accrue interest at 1.5% per month or the legal maximum, whichever is lower.",
        markers=("EVAL-MSA-3.2",),
        required_answer_terms=("1.5%", "per month"),
        forbidden_answer_terms=("5%", "30 calendar days"),
    ),
    EvalCase(
        id="eval_010_saas_availability_commitment",
        question="What monthly availability level does the SaaS provider commit to?",
        answer_type="answerable",
        gold_answer="Provider commits to 99.9% monthly production service availability.",
        markers=("EVAL-MSA-4.1",),
        required_answer_terms=("99.9%", "calendar month"),
        forbidden_answer_terms=("99.5%", "weekly"),
    ),
    EvalCase(
        id="eval_011_saas_service_credit",
        question="What is the service credit if SaaS availability falls below the commitment?",
        answer_type="answerable",
        gold_answer="The sole remedy is a service credit equal to 5% of the affected monthly subscription fee.",
        markers=("EVAL-MSA-4.1",),
        required_answer_terms=("5%", "monthly subscription fee"),
        forbidden_answer_terms=("10%", "refund of all fees"),
    ),
    EvalCase(
        id="eval_012_saas_liability_cap",
        question="What is the general liability cap in the SaaS MSA?",
        answer_type="answerable",
        gold_answer="The general liability cap is the fees paid or payable in the 12 months before the claim event.",
        markers=("EVAL-MSA-7.3",),
        required_answer_terms=("12 months", "fees paid or payable"),
        forbidden_answer_terms=("six months", "unlimited"),
    ),
    EvalCase(
        id="eval_013_saas_data_breach_cap",
        question="What special cap applies to unauthorized disclosure of Customer Data?",
        answer_type="answerable",
        gold_answer="For unauthorized disclosure of Customer Data, Provider's cap is two times the fees paid or payable in the same 12 month period.",
        markers=("EVAL-MSA-7.3",),
        required_answer_terms=("two times", "Customer Data"),
        forbidden_answer_terms=("one times", "five times"),
    ),
    EvalCase(
        id="eval_014_saas_cure_period",
        question="How long is the cure period for material breach under the SaaS MSA?",
        answer_type="answerable",
        gold_answer="The cure period is 60 days after written notice.",
        markers=("EVAL-MSA-9.2",),
        required_answer_terms=("60 days", "written notice"),
        forbidden_answer_terms=("30 days", "15 days"),
    ),
    EvalCase(
        id="eval_015_saas_data_export_window",
        question="How long after termination will Customer Data remain available for export?",
        answer_type="answerable",
        gold_answer="Customer Data will be available for export for 30 days after termination.",
        markers=("EVAL-MSA-11.4",),
        required_answer_terms=("30 days", "CSV or JSON"),
        forbidden_answer_terms=("45 days", "PDF only"),
    ),
    EvalCase(
        id="eval_016_saas_source_code_escrow_refusal",
        question="Does the SaaS MSA require source code escrow?",
        answer_type="unanswerable",
        gold_answer="The indexed SaaS MSA does not state a source code escrow requirement.",
    ),
    EvalCase(
        id="eval_017_dpa_processing_instructions",
        question="On whose instructions may the processor process Personal Data?",
        answer_type="answerable",
        gold_answer="Processor may process Personal Data only on documented instructions from Controller.",
        markers=("EVAL-DPA-2.1",),
        required_answer_terms=("documented instructions", "Controller"),
        forbidden_answer_terms=("Processor's discretion",),
    ),
    EvalCase(
        id="eval_018_dpa_encryption_requirements",
        question="What encryption standards must the processor use for Personal Data?",
        answer_type="answerable",
        gold_answer="Personal Data must be encrypted at rest using AES-256 or stronger and in transit using TLS 1.2 or higher.",
        markers=("EVAL-DPA-4.2",),
        required_answer_terms=("AES-256", "TLS 1.2"),
        forbidden_answer_terms=("TLS 1.0", "DES"),
    ),
    EvalCase(
        id="eval_019_dpa_subprocessor_notice",
        question="How much prior notice is required before appointing subprocessors?",
        answer_type="answerable",
        gold_answer="Processor must give Controller at least 15 days' prior notice.",
        markers=("EVAL-DPA-5.1",),
        required_answer_terms=("15 days", "prior notice"),
        forbidden_answer_terms=("48 hours", "30 days"),
    ),
    EvalCase(
        id="eval_020_dpa_security_incident_notice",
        question="How quickly must Processor notify Controller after confirming a Security Incident?",
        answer_type="answerable",
        gold_answer="Processor must notify Controller no later than 48 hours after confirming the incident.",
        markers=("EVAL-DPA-6.3",),
        required_answer_terms=("48 hours", "confirming"),
        forbidden_answer_terms=("24 hours", "seven days"),
    ),
    EvalCase(
        id="eval_021_dpa_deletion_period",
        question="When must Personal Data be returned or deleted after termination?",
        answer_type="answerable",
        gold_answer="Processor must return or delete Personal Data within 45 days unless law requires retention.",
        markers=("EVAL-DPA-8.1",),
        required_answer_terms=("45 days", "retention"),
        forbidden_answer_terms=("10 business days", "one year"),
    ),
    EvalCase(
        id="eval_022_dpa_data_residency_refusal",
        question="Does the DPA require Personal Data to remain only in Germany?",
        answer_type="unanswerable",
        gold_answer="The indexed DPA does not specify a Germany-only data residency requirement.",
    ),
    EvalCase(
        id="eval_023_procurement_quote_threshold",
        question="How many written quotes are required for purchases over USD 25,000?",
        answer_type="answerable",
        gold_answer="Purchases over USD 25,000 require at least three written quotes unless an approved sole-source justification applies.",
        markers=("EVAL-PROC-4.1",),
        required_answer_terms=("three written quotes", "USD 25,000"),
        forbidden_answer_terms=("two written quotes", "USD 10,000"),
    ),
    EvalCase(
        id="eval_024_procurement_cfo_approval",
        question="What additional approval is required for purchases over USD 100,000?",
        answer_type="answerable",
        gold_answer="Purchases over USD 100,000 require CFO approval before issuing a purchase order.",
        markers=("EVAL-PROC-4.1",),
        required_answer_terms=("USD 100,000", "CFO approval"),
        forbidden_answer_terms=("CEO approval", "Legal approval only"),
    ),
    EvalCase(
        id="eval_025_procurement_no_splitting",
        question="Can employees split related purchases to avoid approval thresholds?",
        answer_type="answerable",
        gold_answer="No. Employees may not split related purchases into smaller orders to avoid thresholds or quote requirements.",
        markers=("EVAL-PROC-4.4",),
        required_answer_terms=("may not split", "approval thresholds"),
        forbidden_answer_terms=("may split",),
    ),
    EvalCase(
        id="eval_026_procurement_payment_terms",
        question="What are the standard supplier payment terms under the procurement policy?",
        answer_type="answerable",
        gold_answer="Standard supplier payment terms are net 45 days from receipt of a valid invoice.",
        markers=("EVAL-PROC-6.1",),
        required_answer_terms=("net 45 days", "valid invoice"),
        forbidden_answer_terms=("net 30 days",),
    ),
    EvalCase(
        id="eval_027_procurement_saas_security_review",
        question="What must SaaS vendors that store Confidential Information provide?",
        answer_type="answerable",
        gold_answer="They must complete security review and provide a current SOC 2 Type II report or equivalent independent assessment.",
        markers=("EVAL-PROC-7.2",),
        required_answer_terms=("SOC 2 Type II", "security review"),
        forbidden_answer_terms=("ISO 9001",),
    ),
    EvalCase(
        id="eval_028_procurement_quote_threshold_chinese",
        question="采购金额超过 25,000 美元时需要几份书面报价?",
        answer_type="answerable",
        gold_answer="Purchases over USD 25,000 require at least three written quotes.",
        markers=("EVAL-PROC-4.1",),
        required_answer_terms=("three written quotes",),
        forbidden_answer_terms=("two written quotes",),
    ),
    EvalCase(
        id="eval_029_procurement_vendor_diversity_refusal",
        question="Does the procurement policy require a minority-owned vendor preference?",
        answer_type="unanswerable",
        gold_answer="The indexed procurement policy does not state a minority-owned vendor preference requirement.",
    ),
    EvalCase(
        id="eval_030_security_mfa_scope",
        question="For which systems is multi-factor authentication required?",
        answer_type="answerable",
        gold_answer="MFA is required for privileged accounts, remote access, production systems, and applications storing Confidential Information.",
        markers=("EVAL-SEC-3.1",),
        required_answer_terms=("privileged accounts", "remote access", "Confidential Information"),
        forbidden_answer_terms=("optional",),
    ),
    EvalCase(
        id="eval_031_security_encryption_alignment",
        question="What encryption does the information security policy require for Confidential Information?",
        answer_type="answerable",
        gold_answer="Confidential Information must be encrypted at rest with AES-256 or an approved equivalent and in transit with TLS 1.2 or higher.",
        markers=("EVAL-SEC-4.2",),
        required_answer_terms=("AES-256", "TLS 1.2"),
        forbidden_answer_terms=("TLS 1.0",),
    ),
    EvalCase(
        id="eval_032_security_critical_patch_sla",
        question="How quickly must critical internet-facing vulnerabilities be remediated?",
        answer_type="answerable",
        gold_answer="Critical internet-facing vulnerabilities must be remediated or mitigated within seven calendar days after validation.",
        markers=("EVAL-SEC-5.2",),
        required_answer_terms=("seven calendar days", "validation"),
        forbidden_answer_terms=("30 calendar days",),
    ),
    EvalCase(
        id="eval_033_security_log_retention",
        question="How long must production authentication and security event logs be retained?",
        answer_type="answerable",
        gold_answer="Those logs must be retained for at least one year.",
        markers=("EVAL-SEC-6.4",),
        required_answer_terms=("one year", "logs"),
        forbidden_answer_terms=("two years", "30 days"),
    ),
    EvalCase(
        id="eval_034_security_vendor_incident_notice",
        question="How quickly must vendors notify Security of a confirmed or suspected incident?",
        answer_type="answerable",
        gold_answer="Vendors handling Confidential Information must notify Security within 24 hours after discovering a confirmed or suspected incident.",
        markers=("EVAL-SEC-8.3",),
        required_answer_terms=("24 hours", "suspected security incident"),
        forbidden_answer_terms=("48 hours", "seven days"),
    ),
    EvalCase(
        id="eval_035_security_dpa_incident_notice_conflict",
        question="Compare the vendor incident notice period in the security policy with the processor incident notice period in the DPA.",
        answer_type="answerable",
        gold_answer="The security policy requires vendor notice within 24 hours, while the DPA requires processor notice no later than 48 hours after confirming a Security Incident.",
        markers=("EVAL-SEC-8.3", "EVAL-DPA-6.3"),
        required_answer_terms=("24 hours", "48 hours"),
        forbidden_answer_terms=("same period",),
    ),
    EvalCase(
        id="eval_036_security_biometric_policy_refusal",
        question="Does the information security policy specify biometric data retention rules?",
        answer_type="unanswerable",
        gold_answer="The indexed information security policy does not specify biometric data retention rules.",
    ),
    EvalCase(
        id="eval_037_hr_remote_work_days",
        question="How many days per week may eligible employees work remotely?",
        answer_type="answerable",
        gold_answer="Eligible employees may work remotely up to three days per week with manager approval.",
        markers=("EVAL-HR-2.1",),
        required_answer_terms=("three days", "manager approval"),
        forbidden_answer_terms=("five days",),
    ),
    EvalCase(
        id="eval_038_hr_pto_accrual",
        question="How much paid time off do full-time employees accrue per calendar year?",
        answer_type="answerable",
        gold_answer="Full-time employees accrue 15 days of paid time off per calendar year.",
        markers=("EVAL-HR-4.3",),
        required_answer_terms=("15 days", "paid time off"),
        forbidden_answer_terms=("20 days",),
    ),
    EvalCase(
        id="eval_039_hr_overtime_approval",
        question="What approval is required before non-exempt employees work overtime?",
        answer_type="answerable",
        gold_answer="Non-exempt employees must obtain manager approval before working overtime, except in emergencies.",
        markers=("EVAL-HR-5.2",),
        required_answer_terms=("manager approval", "overtime"),
        forbidden_answer_terms=("Finance approval",),
    ),
    EvalCase(
        id="eval_040_hr_expense_deadline",
        question="When must employees submit reimbursement requests with receipts?",
        answer_type="answerable",
        gold_answer="Employees must submit reimbursement requests with receipts within 30 days after incurring the business expense.",
        markers=("EVAL-HR-6.1",),
        required_answer_terms=("30 days", "receipts"),
        forbidden_answer_terms=("45 days",),
    ),
    EvalCase(
        id="eval_041_hr_union_refusal",
        question="Does the employee handbook describe a collective bargaining agreement?",
        answer_type="unanswerable",
        gold_answer="The indexed employee handbook excerpt does not describe a collective bargaining agreement.",
    ),
    EvalCase(
        id="eval_042_nda_confidentiality_term",
        question="How long do confidentiality obligations continue under the NDA?",
        answer_type="answerable",
        gold_answer="Confidentiality obligations continue for five years after disclosure, while trade secrets remain protected as long as they qualify.",
        markers=("EVAL-NDA-4.1",),
        required_answer_terms=("five years", "trade secrets"),
        forbidden_answer_terms=("three years",),
    ),
    EvalCase(
        id="eval_043_nda_return_deadline",
        question="How quickly must Confidential Information be returned or destroyed after written request?",
        answer_type="answerable",
        gold_answer="The receiving party must return or destroy Confidential Information within 10 business days after written request.",
        markers=("EVAL-NDA-6.2",),
        required_answer_terms=("10 business days", "written request"),
        forbidden_answer_terms=("30 days",),
    ),
    EvalCase(
        id="eval_044_nda_archival_copy",
        question="May the receiving party retain an archival copy after returning or destroying Confidential Information?",
        answer_type="answerable",
        gold_answer="Yes. One archival copy may be retained solely for legal compliance and backup purposes, subject to continuing confidentiality obligations.",
        markers=("EVAL-NDA-6.2",),
        required_answer_terms=("One archival copy", "legal compliance"),
        forbidden_answer_terms=("unlimited copies",),
    ),
    EvalCase(
        id="eval_045_nda_equitable_relief",
        question="What remedy may the disclosing party seek for unauthorized disclosure?",
        answer_type="answerable",
        gold_answer="The disclosing party may seek injunctive or other equitable relief without posting bond where permitted by law.",
        markers=("EVAL-NDA-8.1",),
        required_answer_terms=("injunctive", "equitable relief"),
        forbidden_answer_terms=("liquidated damages",),
    ),
    EvalCase(
        id="eval_046_nda_non_compete_refusal",
        question="Does the NDA include a non-compete covenant?",
        answer_type="unanswerable",
        gold_answer="The indexed NDA does not include a non-compete covenant.",
    ),
    EvalCase(
        id="eval_047_cross_doc_encryption_alignment",
        question="Compare the DPA encryption requirement with the information security policy encryption requirement.",
        answer_type="answerable",
        gold_answer="Both require AES-256 or stronger/equivalent encryption at rest and TLS 1.2 or higher for data in transit.",
        markers=("EVAL-DPA-4.2", "EVAL-SEC-4.2"),
        required_answer_terms=("AES-256", "TLS 1.2"),
        forbidden_answer_terms=("conflict", "TLS 1.0"),
    ),
    EvalCase(
        id="eval_048_cross_doc_payment_term_difference",
        question="Compare the supply contract invoice payment period with the procurement policy standard supplier payment terms.",
        answer_type="answerable",
        gold_answer="The supply contract requires payment within 30 calendar days after a valid invoice, while the procurement policy standard is net 45 days.",
        markers=("EVAL-C-5.1", "EVAL-PROC-6.1"),
        required_answer_terms=("30 calendar days", "net 45 days"),
        forbidden_answer_terms=("same payment term",),
    ),
]


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    document_paths = []
    for document in DOCUMENTS:
        path = FIXTURE_DIR / document.file_name
        _write_simple_pdf(path, document.pages)
        document_paths.append(path)

    marker_sources = _find_marker_sources(
        document_paths,
        sorted({marker for case in CASES for marker in case.markers}),
    )

    dataset = {
        "version": "0.2",
        "description": (
            "Expanded synthetic legal RAG evaluation dataset. Documents are fictitious "
            "but structured to resemble realistic contracts, policies, and legal work product."
        ),
        "documents": [
            {
                "file_name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_id": file_sha256(path),
            }
            for path in document_paths
        ],
        "cases": [_case_to_dict(case, marker_sources) for case in CASES],
    }

    DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(document_paths)} PDF fixtures to {FIXTURE_DIR}")
    print(f"Wrote {len(CASES)} eval cases to {DATASET_PATH}")


def _case_to_dict(case: EvalCase, marker_sources: dict[str, dict[str, object]]) -> dict[str, object]:
    data: dict[str, object] = {
        "id": case.id,
        "question": case.question,
        "answer_type": case.answer_type,
        "gold_answer": case.gold_answer,
        "gold_sources": [marker_sources[marker] for marker in case.markers],
    }
    if case.answer_type == "unanswerable":
        data["required_refusal_terms"] = list(case.required_refusal_terms or REFUSAL_TERMS)
        return data
    if case.required_answer_terms:
        data["required_answer_terms"] = list(case.required_answer_terms)
    if case.forbidden_answer_terms:
        data["forbidden_answer_terms"] = list(case.forbidden_answer_terms)
    return data


def _find_marker_sources(
    pdf_paths: list[Path],
    markers: list[str],
) -> dict[str, dict[str, object]]:
    remaining = set(markers)
    found: dict[str, dict[str, object]] = {}
    for pdf_path in pdf_paths:
        chunks = _split_like_ingestion(pdf_path)
        for chunk_id, chunk in enumerate(chunks):
            text = chunk.page_content or ""
            for marker in list(remaining):
                if marker not in text:
                    continue
                metadata = chunk.metadata or {}
                found[marker] = {
                    "file_name": pdf_path.name,
                    "page": metadata.get("page"),
                    "chunk_id": chunk_id,
                    "marker": marker,
                }
                remaining.remove(marker)
        if not remaining:
            break

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(f"Could not find generated eval markers in chunks: {missing}")
    return found


def _split_like_ingestion(pdf_path: Path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n第",
            "\nSection ",
            "\nArticle ",
            "\nClause ",
            "\nSchedule ",
            "\nExhibit ",
            "\n\n",
            "\n",
            "。 ",
            "；",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(_split_legal_sections(load_documents(pdf_path)))
    for chunk in chunks:
        chunk.page_content = _chunk_text_with_heading(
            chunk.page_content,
            chunk.metadata.get("section_heading"),
        )
    return chunks


def _write_simple_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []
    page_object_numbers: list[int] = []
    font_object_number = 3 + len(pages) * 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")

    for page_index, lines in enumerate(pages):
        page_object_number = 3 + page_index * 2
        content_object_number = page_object_number + 1
        page_object_numbers.append(page_object_number)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )
        stream = _page_stream(lines)
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("ascii")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    _write_pdf_objects(path, objects)


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 10 Tf", "48 750 Td"]
    first_line = True
    for raw_line in lines:
        wrapped_lines = wrap(raw_line, width=100) if raw_line else [""]
        for line in wrapped_lines:
            if first_line:
                first_line = False
            else:
                commands.append("0 -14 Td")
            commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


if __name__ == "__main__":
    main()
