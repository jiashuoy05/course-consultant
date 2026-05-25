import json
import os
from typing import List, Dict, Any

from langchain_core.documents import Document


def load_admin_data(directory: str) -> List[Document]:
    """Load administrative data from .txt and .md files."""
    print(f"Loading admin data from {directory}...")
    documents: List[Document] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": os.path.basename(file_path), "type": "administrative"},
                    ))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents


def _format_course_as_text(course: Dict[str, Any], source: str) -> str:
    """Format a course dictionary into readable text for embedding."""
    parts = []

    # Course code and name
    code = course.get('code', '')
    name = course.get('name', {})
    if isinstance(name, dict):
        name_zh = name.get('zh', '')
        name_en = name.get('en', '')
        name_str = f"{name_zh} ({name_en})" if name_en else name_zh
    else:
        name_str = str(name)

    if code:
        parts.append(f"課程代碼: {code}")
    if name_str:
        parts.append(f"課程名稱: {name_str}")

    # Description
    description = course.get('description', {})
    if isinstance(description, dict):
        desc_zh = description.get('zh', '')
        if desc_zh:
            parts.append(f"課程描述: {desc_zh}")
    elif description:
        parts.append(f"課程描述: {description}")

    # Credits and hours
    credit = course.get('credit', '')
    hours = course.get('hours', '')
    if credit:
        parts.append(f"學分: {credit}")
    if hours:
        parts.append(f"時數: {hours}")

    # Teachers
    teachers = course.get('teachers', [])
    if teachers:
        parts.append(f"授課教師: {', '.join(teachers)}")

    # Class names
    class_names = course.get('class_names', [])
    if class_names:
        parts.append(f"開課系所: {', '.join(class_names)}")

    # Notes
    notes = course.get('notes', '')
    if notes:
        parts.append(f"備註: {notes}")

    # Join all parts
    text = " | ".join(parts)
    return text


def _format_syllabus_as_text(syllabus: Dict[str, Any], course_id: str) -> str:
    """Format a course syllabus into readable text for embedding."""
    parts = []

    parts.append(f"課程代碼: {course_id}")

    # Instructor
    instructor = syllabus.get('instructor_name', '')
    if instructor:
        parts.append(f"授課教師: {instructor}")

    # Objective
    objective = syllabus.get('objective', '')
    if objective:
        parts.append(f"課程目標: {objective}")

    # Schedule
    schedule = syllabus.get('schedule', '')
    if schedule:
        parts.append(f"課程進度: {schedule}")

    # Score policy
    score_policy = syllabus.get('score_policy', '')
    if score_policy:
        parts.append(f"成績評定: {score_policy}")

    # Materials
    materials = syllabus.get('materials', '')
    if materials:
        parts.append(f"教材: {materials}")

    # Consultation
    consultation = syllabus.get('consultation', '')
    if consultation:
        parts.append(f"諮詢時間: {consultation}")

    # SDG and AI integration
    sdg = syllabus.get('sdg_indicators', '')
    if sdg and sdg != '無（None）':
        parts.append(f"SDG指標: {sdg}")

    ai = syllabus.get('ai_integration', '')
    if ai and ai != '● 無（None）':
        parts.append(f"AI應用: {ai}")

    text = " | ".join(parts)
    return text


def _format_graduation_rule_as_text(rule: Dict[str, Any]) -> str:
    """Format a graduation rule dictionary into readable text for embedding."""
    parts = []
    if rule.get('degree_type'):
        parts.append(f"學制: {rule['degree_type']}")
    if rule.get('department'):
        parts.append(f"系所: {rule['department']}")
    if rule.get('credits'):
        parts.append(f"畢業學分規定: {rule['credits']}")
    if rule.get('rules'):
        parts.append(f"畢業規定: {rule['rules']}")
    return " | ".join(parts)


_DEFAULT_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')


def load_preprocessed_courses(processed_dir: str = _DEFAULT_PROCESSED_DIR) -> List[Document]:
    """Load preprocessed course data from the processed directory."""
    print(f"Loading preprocessed courses from {processed_dir}...")
    documents: List[Document] = []

    # Load main courses
    main_courses_file = os.path.join(processed_dir, 'main_courses.json')
    if os.path.exists(main_courses_file):
        print(f"  Loading main courses...")
        with open(main_courses_file, 'r', encoding='utf-8') as f:
            main_courses = json.load(f)

        for course in main_courses:
            text = _format_course_as_text(course, "main_courses.json")
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": "main_courses.json",
                    "type": "course_listing",
                    "code": course.get('code', ''),
                },
            ))

    # Load research courses
    research_courses_file = os.path.join(processed_dir, 'research_courses.json')
    if os.path.exists(research_courses_file):
        print(f"  Loading research courses...")
        with open(research_courses_file, 'r', encoding='utf-8') as f:
            research_courses = json.load(f)

        for course in research_courses:
            text = _format_course_as_text(course, "research_courses.json")
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": "research_courses.json",
                    "type": "course_listing",
                    "code": course.get('code', ''),
                },
            ))

    # Load course syllabi
    syllabi_file = os.path.join(processed_dir, 'course_syllabi.json')
    if os.path.exists(syllabi_file):
        print(f"  Loading course syllabi...")
        with open(syllabi_file, 'r', encoding='utf-8') as f:
            syllabi = json.load(f)

        for course_id, entries in syllabi.items():
            for syllabus in entries:
                text = _format_syllabus_as_text(syllabus, course_id)
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": "course_syllabi.json",
                        "type": "course_syllabus",
                        "course_id": course_id,
                    },
                ))

    # Load graduation rules (from standard.json processing)
    graduation_rules_file = os.path.join(processed_dir, 'graduation_rules.json')
    if os.path.exists(graduation_rules_file):
        print(f"  Loading graduation rules...")
        with open(graduation_rules_file, 'r', encoding='utf-8') as f:
            graduation_rules = json.load(f)

        for rule in graduation_rules:
            text = _format_graduation_rule_as_text(rule)
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": "graduation_rules.json",
                    "type": "graduation_rule",
                    "degree_type": rule.get('degree_type', ''),
                    "department": rule.get('department', ''),
                },
            ))

    print(f"  Loaded total {len(documents)} documents")
    return documents
