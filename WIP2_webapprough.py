# from pyresparser import ResumeParser
# from docx import Document
# from flask import Flask,render_template,redirect,request
import numpy as np
import pandas as pd
# import re
# from ftfy import fix_text
import os
from resume_parser import resumeparse

import streamlit as st
import PyPDF2
# from functions import convert_pdf_to_txt_file

fl = st.file_uploader("Upload a resume PDF file")
# if fl:
    # st.write(fl.name)
    # strg = StringIO(fl.getvalue().decode("utf-8"))
    # st.write(strg)
if fl:
    pdfobj = open(fl.name,'rb')
    reader = PyPDF2.PdfReader(pdfobj)
    txt = str()
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        txt = txt + page.extract_text()
    # page = reader.pages[0]
    # st.write(page.extract_text())
    #st.write(txt)
    with open("testing.txt",'w',encoding="utf-8") as f:
        f.write(txt)
    # data = ResumeParser("testing.txt").get_extracted_data()
    data = resumeparse.read_file("testing.txt")
    skills = data['skills']
    st.write(skills)


# working
# if fl:
#     pdfobj = open(fl.name,'rb')
#     pdfr = PyPDF2.PdfFileReader(pdfobj)
#     x = pdfr.numPages
#     # st.write(x)
#     pgobj = pdfr.getPages()
#     tex_dat = pgobj.extractText()
#     st.write(tex_dat)
#     # text_data_f = functions.convert_pdf_to_txt_file(fl)
#     data = ResumeParser(tex_dat).get_extracted_data()
#     resume = data['skills']
#     print(type(resume))




# if fl is not None:
#     try:
#         doc = Document()
#         with open(fl.name, 'rb') as input:
#             # st.text(input.read())
#             doc.add_paragraph(input.read())
#             doc.save("text.docx")
#             data = ResumeParser('text.docx').get_extracted_data()
#     except FileNotFoundError:
#         st.error('File not found.')
#         data = ResumeParser(f.filename).get_extracted_data()
#     resume=data['skills']
#     print(type(resume))

# if fl is not None:
#     doc = Document()
#     f = fl.read()
#     # with open(f, 'r') as file:
#     st.write(f)
    # doc.add_paragraph(f)
    # doc.save("text.docx")
    # data = ResumeParser('text.docx').get_extracted_data()

    # resume=data['skills']
    # print(type(resume))

    # skills=[]
    # skills.append(' '.join(word for word in resume))
    # org_name_clean = skills
###########################################################
# f = request.files['userfile']
# if f is not None:
#     f.save(f.filename)
#     try:
#         doc = Document()
#         with open(f.filename, 'r') as file:
#             doc.add_paragraph(file.read())
#             doc.save("text.docx")
#             data = ResumeParser('text.docx').get_extracted_data()

#     except:
#         data = ResumeParser(f.filename).get_extracted_data()
#     resume=data['skills']
#     print(type(resume))

#     skills=[]
#     skills.append(' '.join(word for word in resume))
#     org_name_clean = skills