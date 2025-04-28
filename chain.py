import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama3-70b-8192"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, user_info):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### USER INFO:
            Name: {name}
            Email: {email}
            Company: {company}
            Designation: {designation}
            Experience: {experience}
            Skills: {skills}

            ### INSTRUCTION:
            Write a cold email for the job that:
            1. Explicitly matches the user's qualifications to job requirements
            2. Emphasizes relevant technical skills and projects
            3. Highlights education status and graduation timeline if relevant
            4. Maintains professional but enthusiastic tone
            5. Uses specific examples from projects to demonstrate capabilities
            6. Only includes information provided above
            7. Shows a positive attitude if the candidate’s skills match the job description.

            Remember: The email must be 100% truthful and based only on the provided information.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            **user_info,
            "skills": user_info.get("skills", [])
        })
        return res.content
