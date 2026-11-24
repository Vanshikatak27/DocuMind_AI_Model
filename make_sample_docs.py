"""
make_sample_docs.py
--------------------
Generates a small set of realistic sample academic PDFs so DocuMind AI has
something to ingest out of the box (a real deployment would just point the
loader at the student's own college documents instead).

Run:  python3 make_sample_docs.py
"""
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceAfter=8, spaceBefore=14)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=10, leading=15)


def write_pdf(filename, title, sections):
    """sections: list of (heading, [paragraphs]) tuples, each heading starts a fresh page-friendly block."""
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                             leftMargin=0.9 * inch, rightMargin=0.9 * inch)
    story = [Paragraph(title, H1), Spacer(1, 0.15 * inch)]
    for heading, paras in sections:
        story.append(Paragraph(heading, H2))
        for p in paras:
            story.append(Paragraph(p, BODY))
    doc.build(story)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. GenAI Implementation Guide  (the document most demo questions target)
# ---------------------------------------------------------------------------
write_pdf(
    "GenAI_Implementation_Guide.pdf",
    "Generative AI — Model Making: Implementation Guide",
    [
        ("Overview of the Model Making Track", [
            "The Model Making track requires every group to design, build and demonstrate a working "
            "Generative AI system across two internal juries (Jury 1 and Jury 2) followed by an "
            "End Semester Examination (ESE) jury. The same project topic is carried through all three "
            "evaluation stages so that the work can be progressively deepened rather than restarted.",
        ]),
        ("Jury 1 Requirements", [
            "Jury 1 is the first presentation of the selected topic. Groups must submit a problem statement, "
            "a proposed Generative AI solution, the intended architecture, and a basic working prototype "
            "covering at least one end-to-end pass of the pipeline (for example: input, processing and a "
            "generated output). A live demonstration of the prototype is mandatory for Jury 1.",
        ]),
        ("Jury 2 Requirements", [
            "Model Making groups continue with the same topic for Jury 2 and extend and deepen the model "
            "developed for Jury 1. Jury 2 expects improved accuracy or robustness, additional features "
            "identified as future work in Jury 1, and a more complete evaluation of the system on a larger "
            "set of test cases. Changing the core topic between Jury 1 and Jury 2 is not permitted.",
        ]),
        ("Can the Topic Change After Jury 1?", [
            "No. For the Model Making category, the topic finalized at Jury 1 must be carried forward to "
            "Jury 2 and the ESE jury. Only the depth, features and evaluation rigor are expected to increase; "
            "the underlying problem and Generative AI approach should remain the same.",
        ]),
        ("ESE Jury Requirements", [
            "The End Semester Examination (ESE) jury requires substantial and in-depth work on the selected "
            "topic, going beyond what was shown at Jury 2. Groups are expected to present a polished, "
            "interactive application, a complete evaluation covering accuracy and reliability metrics, and a "
            "clear articulation of the underlying Generative AI concepts to the panel. Every group member "
            "must be able to answer questions about the section of the project they were responsible for.",
        ]),
        ("Recommended Technology Stack", [
            "For Retrieval-Augmented Generation style Model Making projects, the recommended stack is Python "
            "developed in Google Colab, PyPDF for document parsing, Sentence Transformers for embeddings, "
            "FAISS as the vector database, an open or API-based Large Language Model for generation, and "
            "Gradio for the final interactive interface.",
        ]),
        ("Group Work Division", [
            "Groups of four are expected to divide responsibilities across document processing, embeddings "
            "and retrieval, LLM and prompt engineering, and interface plus evaluation, while every member "
            "retains a working understanding of the complete pipeline.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 2. Academic Rules & Regulations
# ---------------------------------------------------------------------------
write_pdf(
    "Academic_Rules_and_Regulations.pdf",
    "Academic Rules and Regulations",
    [
        ("Attendance Requirement", [
            "A minimum of 75% attendance is required in each course, including lectures, practicals and "
            "tutorials, for a student to be eligible to appear for the End Semester Examination. Students "
            "with attendance between 65% and 75% may be granted condonation only with valid medical or "
            "documented grounds, subject to approval by the department committee.",
        ]),
        ("Examination Eligibility", [
            "Students must clear all internal assessments with a minimum aggregate of 40% to be eligible for "
            "the End Semester Examination. Backlogs from previous semesters must be cleared as per the "
            "academic calendar published by the examination cell.",
        ]),
        ("Grading Policy", [
            "The university follows a ten-point grading scale. Final grades are computed from a weighted "
            "combination of internal assessment, mid-semester evaluation and the End Semester Examination, "
            "with weightages published separately for theory and laboratory courses.",
        ]),
        ("Malpractice and Discipline", [
            "Any form of academic malpractice during internal assessments, project evaluations or "
            "examinations, including plagiarism in project reports, will be referred to the Unfair Means "
            "Committee and may result in penalties ranging from a warning to cancellation of the examination.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 3. Project & Model Making Guidelines
# ---------------------------------------------------------------------------
write_pdf(
    "Project_Guidelines.pdf",
    "Final Year Project and Model Making Guidelines",
    [
        ("Project Categories", [
            "Final-year Generative AI projects may be undertaken under one of three categories: Model "
            "Making, Real-Time Scenario, or Case Study. Each category has a distinct evaluation rubric and "
            "distinct expectations for continuity of topic across juries.",
        ]),
        ("Model Making Category", [
            "In the Model Making category, groups build and demonstrate an actual working Generative AI "
            "system. The same topic and system are carried through Jury 1, Jury 2 and the ESE jury, with "
            "each stage expected to add depth, robustness or new capabilities to the same underlying model.",
        ]),
        ("Real-Time Scenario Category", [
            "Groups in the Real-Time Scenario category continue with the same topic across juries as well, "
            "applying their Generative AI solution to a live or simulated real-world scenario and "
            "demonstrating its behaviour under realistic conditions.",
        ]),
        ("Case Study Category", [
            "The Case Study category requires groups to select a new topic for the ESE stage, focusing on "
            "an in-depth analysis and comparison of existing Generative AI approaches rather than continuing "
            "a single built system.",
        ]),
        ("Group Size and Formation", [
            "Projects are undertaken in groups of four students. Group formation must be finalized before "
            "the topic proposal is submitted, and changes to group composition after Jury 1 require written "
            "approval from the project coordinator.",
        ]),
        ("Report Submission", [
            "Each group must submit a project report following the department template after every jury, "
            "documenting the problem statement, methodology, architecture, implementation and evaluation "
            "results relevant to that stage.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 4. Placement Guidelines
# ---------------------------------------------------------------------------
write_pdf(
    "Placement_Guidelines.pdf",
    "Placement Cell — Guidelines for Final Year Students",
    [
        ("Eligibility Criteria", [
            "Students must have a minimum aggregate CGPA of 6.0 with no active backlogs at the time of "
            "registration to be eligible for on-campus placement drives. Some recruiters may specify a "
            "higher CGPA cutoff, which will be published along with the job description.",
        ]),
        ("Registration Process", [
            "Students must register on the placement portal at the start of the final year and keep their "
            "resume and academic details updated before each placement season begins.",
        ]),
        ("Placement Rounds", [
            "A typical placement process includes an online aptitude test, a technical interview, and a "
            "human resources interview. Some companies additionally conduct a group discussion or a coding "
            "round before shortlisting candidates for interviews.",
        ]),
        ("Offer Policy", [
            "Students who accept a placement offer through the campus process are expected to honour that "
            "commitment and are generally not permitted to sit for further placement drives in the same "
            "academic year, except for a higher package dream-company drive as defined by placement policy.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 5. Student Handbook
# ---------------------------------------------------------------------------
write_pdf(
    "Student_Handbook.pdf",
    "Student Handbook",
    [
        ("Code of Conduct", [
            "Students are expected to maintain discipline and integrity on campus, including during "
            "project work, laboratory sessions and examinations. Ragging in any form is strictly prohibited "
            "and punishable under university and legal provisions.",
        ]),
        ("Library Facilities", [
            "The central library provides access to technical journals, e-books and a dedicated Generative "
            "AI and Data Science reading section. Final-year project groups may reserve discussion rooms for "
            "project meetings during working hours.",
        ]),
        ("Grievance Redressal", [
            "Students may raise academic or administrative grievances, including disputes related to project "
            "evaluation, through the departmental grievance redressal committee within two weeks of the "
            "concerned event.",
        ]),
        ("Extracurricular Participation", [
            "Students are encouraged to participate in technical fests, hackathons and Generative AI "
            "workshops organized by the department, subject to maintaining the minimum attendance "
            "requirement in regular coursework.",
        ]),
    ],
)

print("\nAll sample documents generated in:", OUT_DIR)
