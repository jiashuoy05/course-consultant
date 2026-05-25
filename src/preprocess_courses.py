"""
Preprocessing script for course data from data/114/
Extracts meaningful content and removes:
- Internal JSP links (navigation noise)
- Numeric IDs used for relational references
"""

import json
import os
import re
from typing import List, Dict, Any


def remove_jsp_links(text: str) -> str:
    """Remove internal JSP link patterns from text."""
    if not text:
        return text

    # Remove JSP links like: Subj.jsp?format=-4&year=114&sem=1&code=2652
    text = re.sub(r'(https?://)?[\w\-\.]+\.jsp\?[^\s,\}\]]*', '', text)
    return text


def clean_text_field(text: str) -> str:
    """Clean a text field by removing JSP links and extra whitespace."""
    if not isinstance(text, str):
        return text
    text = remove_jsp_links(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def extract_course_from_main(course: Dict) -> Dict[str, Any]:
    """Extract relevant fields from a course in main.json."""
    extracted = {
        'code': course.get('code'),
        'name': course.get('name'),
        'description': course.get('description'),
        'credit': course.get('credit'),
        'hours': course.get('hours'),
        'stage': course.get('stage'),
        'courseType': course.get('courseType'),
        'teachers': [teacher.get('name') for teacher in course.get('teacher', [])],
        'class_names': [cls.get('name') for cls in course.get('class', [])],
        'notes': course.get('notes'),
        'language': course.get('language'),
        'people': course.get('people'),
    }

    # Clean text fields
    if isinstance(extracted['description'], dict):
        extracted['description'] = {
            lang: clean_text_field(text)
            for lang, text in extracted['description'].items()
        }
    if isinstance(extracted['name'], dict):
        extracted['name'] = {
            lang: clean_text_field(text)
            for lang, text in extracted['name'].items()
        }
    if extracted['notes']:
        extracted['notes'] = clean_text_field(extracted['notes'])

    return extracted


def process_main_json(file_path: str) -> List[Dict[str, Any]]:
    """Process main.json - hierarchical course structure."""
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    courses = []
    for course in data:
        extracted = extract_course_from_main(course)
        courses.append(extracted)

    print(f"  Extracted {len(courses)} courses from {file_path}")
    return courses


def extract_syllabus_from_course_file(data: List[Dict]) -> List[Dict[str, Any]]:
    """Extract syllabus information from a course file.
    Returns a list since courses may have multiple instructor sections.
    """
    if not data:
        return []

    results = []
    for instructor_data in data:
        extracted = {
            'instructor_name': instructor_data.get('name'),
            'email': instructor_data.get('email'),
            'objective': clean_text_field(instructor_data.get('objective', '')),
            'schedule': clean_text_field(instructor_data.get('schedule', '')),
            'score_policy': clean_text_field(instructor_data.get('scorePolicy', '')),
            'materials': clean_text_field(instructor_data.get('materials', '')),
            'consultation': clean_text_field(instructor_data.get('consultation', '')),
            'remarks': clean_text_field(instructor_data.get('remarks', '')),
            'sdg_indicators': instructor_data.get('課程對應SDGs指標', ''),
            'ai_integration': instructor_data.get('課程是否導入AI', ''),
            'latest_update': instructor_data.get('latestUpdate'),
        }
        results.append(extracted)

    return results


def process_course_files(directory: str) -> Dict[str, Dict[str, Any]]:
    """Process all course files in the course directory."""
    print(f"Processing course files from {directory}...")

    courses_data = {}
    file_count = 0
    error_count = 0

    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                course_id = file_name.replace('.json', '')
                extracted_list = extract_syllabus_from_course_file(data)
                if extracted_list:
                    courses_data[course_id] = extracted_list
                    file_count += 1

                    if file_count % 500 == 0:
                        print(f"  Processed {file_count} course files...")

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"  Error processing {file_path}: {e}")

    if error_count > 5:
        print(f"  ... and {error_count - 5} more errors")

    print(f"  Processed {file_count} course files ({error_count} errors)")
    return courses_data


def create_id_mapping(main_courses: List[Dict], courses_dir_data: Dict) -> Dict[str, Dict]:
    """
    Create a mapping of course IDs to course information for later reference.
    Useful for linking related documents.
    """
    mapping = {}

    # Map from main.json courses (using code as identifier)
    for course in main_courses:
        code = course.get('code')
        if code:
            mapping[code] = {
                'type': 'course_listing',
                'name': course.get('name'),
                'credit': course.get('credit'),
            }

    # Map from course directory (using file ID as identifier)
    for course_id, entries in courses_dir_data.items():
        primary = entries[0]
        mapping[course_id] = {
            'type': 'course_syllabus',
            'instructor': primary.get('instructor_name'),
            'latest_update': primary.get('latest_update'),
        }

    return mapping


def process_standard_json(file_path: str) -> tuple:
    """Process standard.json — returns (graduation_rules, standard_courses).

    standard.json has fields absent from main.json/research.json:
    - degree_type (大學部 / 碩士班 / 博士班 / etc.) per course
    - year (1st/2nd/3rd/4th) and sem (1/2) per course
    - graduation credit requirements and rules per department
    """
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graduation_rules = []
    standard_courses = []

    for degree_type, departments in data.items():
        for dept_name, content in departments.items():
            # Graduation rules document (one per dept × degree_type)
            credits = content.get('credits', {})
            rules = content.get('rules') or []
            if credits or rules:
                credits_text = ' | '.join(f"{k}: {v}" for k, v in credits.items())
                rules_text = ' '.join(rules)
                graduation_rules.append({
                    'degree_type': degree_type,
                    'department': dept_name,
                    'credits': credits_text,
                    'rules': rules_text,
                })

            # Course entries enriched with degree_type, year, sem
            for course in content.get('courses', []):
                standard_courses.append({
                    'name': course.get('name'),
                    'credit': course.get('credit'),
                    'type': course.get('type'),
                    'year': course.get('year'),
                    'sem': course.get('sem'),
                    'degree_type': degree_type,
                    'department': dept_name,
                })

    print(f"  Extracted {len(graduation_rules)} graduation rules, {len(standard_courses)} course entries")
    return graduation_rules, standard_courses


def preprocess_all(data_dir: str = 'data/114', output_dir: str = 'data/processed'):
    """Main function to preprocess all course data."""

    os.makedirs(output_dir, exist_ok=True)

    # Process main.json
    main_json_path = os.path.join(data_dir, 'main.json')
    if os.path.exists(main_json_path):
        main_courses = process_main_json(main_json_path)
        with open(os.path.join(output_dir, 'main_courses.json'), 'w', encoding='utf-8') as f:
            json.dump(main_courses, f, ensure_ascii=False, indent=2)
    else:
        main_courses = []
        print(f"Warning: {main_json_path} not found")

    # Process research program file
    research_file = os.path.join(data_dir, '研究所(日間部、進修部、週末碩士班).json')
    if os.path.exists(research_file):
        research_courses = process_main_json(research_file)
        with open(os.path.join(output_dir, 'research_courses.json'), 'w', encoding='utf-8') as f:
            json.dump(research_courses, f, ensure_ascii=False, indent=2)
    else:
        research_courses = []
        print(f"Warning: {research_file} not found")

    # Process course directory
    courses_dir = os.path.join(data_dir, 'course')
    if os.path.isdir(courses_dir):
        courses_data = process_course_files(courses_dir)
        with open(os.path.join(output_dir, 'course_syllabi.json'), 'w', encoding='utf-8') as f:
            json.dump(courses_data, f, ensure_ascii=False, indent=2)
    else:
        courses_data = {}
        print(f"Warning: {courses_dir} not found")

    # Process standard.json (graduation rules + degree-type-tagged courses)
    standard_json_path = os.path.join(data_dir, 'standard.json')
    if os.path.exists(standard_json_path):
        graduation_rules, standard_courses = process_standard_json(standard_json_path)
        with open(os.path.join(output_dir, 'graduation_rules.json'), 'w', encoding='utf-8') as f:
            json.dump(graduation_rules, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, 'standard_courses.json'), 'w', encoding='utf-8') as f:
            json.dump(standard_courses, f, ensure_ascii=False, indent=2)
    else:
        graduation_rules, standard_courses = [], []
        print(f"Warning: {standard_json_path} not found")

    # Create ID mapping
    id_mapping = create_id_mapping(main_courses + research_courses, courses_data)
    with open(os.path.join(output_dir, 'id_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "="*60)
    print("Preprocessing Summary:")
    print(f"  Main courses: {len(main_courses)}")
    print(f"  Research courses: {len(research_courses)}")
    print(f"  Course syllabi: {len(courses_data)}")
    print(f"  Graduation rules: {len(graduation_rules)}")
    print(f"  Standard courses (with degree/year/sem): {len(standard_courses)}")
    print(f"  ID mappings: {len(id_mapping)}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    preprocess_all()
