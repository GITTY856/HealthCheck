import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib

class ModernHealthCheckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HEALTHCHECK - AI Disease Prediction")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f8f9fa')
        self.setup_styles()
        self.setup_main_window()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f8f9fa')
        self.style.configure('Nav.TFrame', background='#2c3e50')
        self.style.configure('Card.TFrame', background='white', relief='raised', borderwidth=0)
        self.style.configure('Title.TLabel', background='#2c3e50', foreground='white', font=('Arial', 16, 'bold'))
        self.style.configure('Subtitle.TLabel', background='white', foreground='#2c3e50', font=('Arial', 12, 'bold'))
        self.style.configure('Fact.TLabel', background='#ecf0f1', foreground='#2c3e50', font=('Arial', 10))
        
    def setup_main_window(self):
        self.main_container = ttk.Frame(self.root, style='Modern.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.create_navbar()
        self.create_content_area()
        self.create_footer()
        
    def create_navbar(self):
        navbar = ttk.Frame(self.main_container, style='Nav.TFrame', height=80)
        navbar.pack(fill=tk.X, side=tk.TOP)
        navbar.pack_propagate(False)
        
        title_label = ttk.Label(navbar, text="HEALTHCHECK", style='Title.TLabel', font=('Arial', 20, 'bold'))
        title_label.pack(side=tk.LEFT, padx=30, pady=20)
        
        nav_buttons = ttk.Frame(navbar, style='Nav.TFrame')
        nav_buttons.pack(side=tk.RIGHT, padx=30, pady=20)
        
        nav_items = [("About Us", self.show_about), 
                    ("Medical Terms", self.show_terminology),
                    ("Doctors", self.show_doctors)]
        
        for text, command in nav_items:
            btn = tk.Button(nav_buttons, text=text, bg='#3498db', fg='white',
                          font=('Arial', 10, 'bold'), relief='flat', padx=15, pady=8,
                          command=command)
            btn.pack(side=tk.LEFT, padx=8)
            
    def create_content_area(self):
        content_frame = ttk.Frame(self.main_container, style='Modern.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        left_sidebar = ttk.Frame(content_frame, style='Card.TFrame', width=320)
        left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 25))
        left_sidebar.pack_propagate(False)
        
        self.create_sidebar_content(left_sidebar)
        
        right_content = ttk.Frame(content_frame, style='Modern.TFrame')
        right_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_disease_grid(right_content)
        
    def create_sidebar_content(self, parent):
        header = ttk.Frame(parent, style='Card.TFrame')
        header.pack(fill=tk.X, pady=20, padx=20)
        
        ttk.Label(header, text="Health Insights", font=('Arial', 18, 'bold'), 
                 background='white', foreground='#2c3e50').pack()
        
        insights_frame = ttk.Frame(parent, style='Card.TFrame')
        insights_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        health_facts = [
            {"icon": "💪", "text": "Regular exercise reduces chronic disease risk by 40%"},
            {"icon": "😴", "text": "7-8 hours of sleep improves immune function significantly"},
            {"icon": "🥗", "text": "Balanced diet prevents 80% of heart diseases"},
            {"icon": "🩺", "text": "Annual checkups detect 70% of early stage diseases"},
            {"icon": "💧", "text": "Proper hydration improves kidney function by 30%"},
            {"icon": "🧘", "text": "Stress management reduces stroke risk by 25%"}
        ]
        
        for fact in health_facts:
            fact_frame = ttk.Frame(insights_frame, style='Card.TFrame')
            fact_frame.pack(fill=tk.X, pady=12)
            
            icon_label = ttk.Label(fact_frame, text=fact["icon"], font=('Arial', 16),
                                 background='white')
            icon_label.pack(side=tk.LEFT, padx=(0, 10))
            
            text_label = ttk.Label(fact_frame, text=fact["text"], font=('Arial', 10),
                                 background='white', foreground='#34495e', wraplength=250,
                                 justify=tk.LEFT)
            text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
    def create_disease_grid(self, parent):
        header_frame = ttk.Frame(parent, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        ttk.Label(header_frame, text="AI Disease Prediction Models", 
                 font=('Arial', 28, 'bold'), background='#f8f9fa', 
                 foreground='#2c3e50').pack()
        
        ttk.Label(header_frame, text="Select a disease model to assess your health risk", 
                 font=('Arial', 14), background='#f8f9fa', 
                 foreground='#7f8c8d').pack(pady=5)
        
        grid_frame = ttk.Frame(parent, style='Modern.TFrame')
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        disease_data = [
            {"name": "Diabetes", "icon": "🩸", "color": "#e74c3c", "desc": "Blood sugar level analysis"},
            {"name": "Liver", "icon": "🧬", "color": "#e67e22", "desc": "Liver function assessment"},
            {"name": "COPD", "icon": "🌬️", "color": "#f1c40f", "desc": "Respiratory health evaluation"},
            {"name": "Pancreas", "icon": "🎯", "color": "#2ecc71", "desc": "Pancreatic health screening"},
            {"name": "Heart", "icon": "❤️", "color": "#9b59b6", "desc": "Cardiovascular risk analysis"},
            {"name": "Stroke", "icon": "🧠", "color": "#34495e", "desc": "Stroke risk prediction"},
            {"name": "Kidney", "icon": "💊", "color": "#1abc9c", "desc": "Kidney function assessment"}
        ]
        
        for i, disease in enumerate(disease_data):
            row, col = i // 3, i % 3
            card = self.create_disease_card(grid_frame, disease)
            card.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')
            
        for i in range(3):
            grid_frame.columnconfigure(i, weight=1)
        for i in range(3):
            grid_frame.rowconfigure(i, weight=1)
            
    def create_disease_card(self, parent, disease):
        card = tk.Frame(parent, bg='white', relief='raised', bd=1, 
                       highlightbackground='#e0e0e0', highlightthickness=1)
        
        color_header = tk.Frame(card, bg=disease["color"], height=8)
        color_header.pack(fill=tk.X)
        
        content_frame = tk.Frame(card, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=25)
        
        icon_label = tk.Label(content_frame, text=disease["icon"], font=('Arial', 32),
                            bg='white', fg=disease["color"])
        icon_label.pack(pady=(0, 10))
        
        name_label = tk.Label(content_frame, text=disease["name"], 
                            font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        name_label.pack(pady=(0, 8))
        
        desc_label = tk.Label(content_frame, text=disease["desc"], 
                            font=('Arial', 10), bg='white', fg='#7f8c8d',
                            wraplength=180)
        desc_label.pack(pady=(0, 20))
        
        predict_btn = tk.Button(content_frame, text="Predict Risk", 
                              bg=disease["color"], fg='white',
                              font=('Arial', 11, 'bold'), relief='flat',
                              padx=20, pady=10,
                              command=lambda d=disease["name"]: self.open_disease_model(d))
        predict_btn.pack()
        
        card.bind("<Enter>", lambda e, c=card: self.on_card_hover(c, True))
        card.bind("<Leave>", lambda e, c=card: self.on_card_hover(c, False))
        
        return card
        
    def on_card_hover(self, card, is_hover):
        if is_hover:
            card.configure(relief='solid', bd=2, highlightbackground='#3498db')
        else:
            card.configure(relief='raised', bd=1, highlightbackground='#e0e0e0')
            
    def create_footer(self):
        footer = ttk.Frame(self.main_container, style='Nav.TFrame', height=50)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        ttk.Label(footer, text="© 2024 HEALTHCHECK — All rights reserved.", 
                 foreground='white', background='#2c3e50', 
                 font=('Arial', 10)).pack(pady=15)
        
    def show_about(self):
        self.show_modern_dialog("About HEALTHCHECK", 
                               "HEALTHCHECK is an advanced AI-powered disease prediction platform that provides early risk assessment for various medical conditions using state-of-the-art machine learning models.\n\nOur mission is to make healthcare accessible and preventive through technology.")
    
    def show_terminology(self):
        terms = """• BMI (Body Mass Index): Measure of body fat based on height and weight
• BP (Blood Pressure): Force of blood against artery walls
• COPD: Chronic Obstructive Pulmonary Disease
• CKD: Chronic Kidney Disease
• Glucose: Blood sugar level
• HDL/LDL: Good/Bad cholesterol types
• FEV1: Lung function measurement
• A1C: Average blood sugar over 3 months"""
        self.show_modern_dialog("Medical Terminology", terms)
    
    def show_doctors(self):
        doctors = """• Cardiologist: Heart and blood vessel specialist
• Endocrinologist: Diabetes and hormone specialist
• Pulmonologist: Lung and respiratory specialist
• Nephrologist: Kidney disease specialist
• Hepatologist: Liver disease specialist
• Neurologist: Brain and nervous system specialist
• Gastroenterologist: Digestive system specialist"""
        self.show_modern_dialog("Medical Specialists", doctors)
    
    def show_modern_dialog(self, title, content):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("500x450")
        dialog.configure(bg='white')
        dialog.transient(self.root)
        dialog.grab_set()
        
        header = tk.Frame(dialog, bg='#2c3e50', height=80)
        header.pack(fill=tk.X)
        
        tk.Label(header, text=title, font=('Arial', 18, 'bold'), 
                bg='#2c3e50', fg='white').pack(pady=25)
        
        content_frame = tk.Frame(dialog, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)
        
        text_widget = tk.Text(content_frame, font=('Arial', 11), wrap=tk.WORD,
                            bg='white', fg='#2c3e50', padx=10, pady=10,
                            relief='flat', borderwidth=0)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(dialog, text="Close", bg='#3498db', fg='white',
                 font=('Arial', 11, 'bold'), relief='flat', padx=30, pady=10,
                 command=dialog.destroy).pack(pady=20)
    
    def open_disease_model(self, disease):
        model_classes = {
            "Diabetes": DiabetesPredictor,
            "Liver": LiverPredictor,
            "COPD": COPDPredictor,
            "Pancreas": PancreasPredictor,
            "Heart": HeartPredictor,
            "Stroke": StrokePredictor,
            "Kidney": KidneyPredictor
        }
        
        if disease in model_classes:
            model_classes[disease](self.root)

class ModernPredictor:
    def __init__(self, parent, title, fields):
        self.window = tk.Toplevel(parent)
        self.window.title(f"{title} - HEALTHCHECK")
        self.window.geometry("700x800")
        self.window.configure(bg='#f8f9fa')
        self.fields = fields
        self.entries = {}
        self.create_modern_form()
        
    def create_modern_form(self):
        main_frame = tk.Frame(self.window, bg='#f8f9fa')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        header = tk.Frame(main_frame, bg='#2c3e50', height=100)
        header.pack(fill=tk.X)
        
        tk.Label(header, text=self.window.title().split(' - ')[0], 
                font=('Arial', 22, 'bold'), bg='#2c3e50', fg='white').pack(pady=30)
        
        form_container = tk.Frame(main_frame, bg='white', relief='raised', bd=0)
        form_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30, ipadx=20, ipady=20)
        
        form_frame = tk.Frame(form_container, bg='white')
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        for i, field in enumerate(self.fields):
            field_frame = tk.Frame(form_frame, bg='white')
            field_frame.pack(fill=tk.X, pady=12)
            
            tk.Label(field_frame, text=field, font=('Arial', 12, 'bold'),
                    bg='white', fg='#2c3e50').pack(anchor='w', pady=(0, 5))
            
            entry = tk.Entry(field_frame, font=('Arial', 12), relief='solid',
                           bd=1, highlightbackground='#bdc3c7', highlightthickness=1)
            entry.pack(fill=tk.X, ipady=8)
            self.entries[field] = entry
            
        button_frame = tk.Frame(form_container, bg='white')
        button_frame.pack(fill=tk.X, pady=30)
        
        tk.Button(button_frame, text="🔍 Predict Risk", bg='#27ae60', fg='white',
                 font=('Arial', 14, 'bold'), relief='flat', padx=30, pady=12,
                 command=self.predict).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="🗑️ Clear", bg='#e74c3c', fg='white',
                 font=('Arial', 14, 'bold'), relief='flat', padx=30, pady=12,
                 command=self.clear_form).pack(side=tk.LEFT, padx=10)
        
        self.result_frame = tk.Frame(form_container, bg='#ecf0f1', relief='solid', bd=1)
        self.result_frame.pack(fill=tk.X, pady=20)
        
        self.result_label = tk.Label(self.result_frame, text="Enter your health data above to get prediction", 
                                   font=('Arial', 12), bg='#ecf0f1', fg='#7f8c8d', pady=20)
        self.result_label.pack(fill=tk.X)
        
    def clear_form(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="Enter your health data above to get prediction", fg='#7f8c8d')
        
    def show_result(self, text, is_risk=False):
        color = '#e74c3c' if is_risk else '#27ae60'
        self.result_label.config(text=text, fg=color, font=('Arial', 13, 'bold'))


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernHealthCheckApp(root)
    root.mainloop()