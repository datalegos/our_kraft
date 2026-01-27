// School Management UI JavaScript
class SchoolManagementApp {
    constructor() {
        this.apiBase = '/api';
        this.currentTab = 'departments';
        this.departments = [];
        this.students = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDepartments();
        this.loadStudents();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Modal close
        document.querySelectorAll('.close, .modal').forEach(element => {
            element.addEventListener('click', (e) => {
                if (e.target === element) {
                    this.closeModal();
                }
            });
        });

        // Form submissions
        document.getElementById('departmentForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveDepartment();
        });

        document.getElementById('studentForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveStudent();
        });

        // Add buttons
        document.getElementById('addDepartmentBtn').addEventListener('click', () => {
            this.showDepartmentModal();
        });

        document.getElementById('addStudentBtn').addEventListener('click', () => {
            this.showStudentModal();
        });
    }

    switchTab(tabName) {
        // Update nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');

        this.currentTab = tabName;
    }

    async loadDepartments() {
        try {
            this.showLoading('departmentsLoading');
            const response = await fetch(`${this.apiBase}/departments/`);
            this.departments = await response.json();
            this.renderDepartments();
            this.updateDepartmentSelect();
        } catch (error) {
            this.showAlert('Error loading departments: ' + error.message, 'error');
        } finally {
            this.hideLoading('departmentsLoading');
        }
    }

    async loadStudents() {
        try {
            this.showLoading('studentsLoading');
            const response = await fetch(`${this.apiBase}/students/`);
            this.students = await response.json();
            this.renderStudents();
        } catch (error) {
            this.showAlert('Error loading students: ' + error.message, 'error');
        } finally {
            this.hideLoading('studentsLoading');
        }
    }

    renderDepartments() {
        const tbody = document.getElementById('departmentsTableBody');
        
        if (this.departments.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="4" class="empty-state">
                        <h3>No departments found</h3>
                        <p>Click "Add Department" to create your first department</p>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = this.departments.map(dept => `
            <tr>
                <td><strong>${dept.name}</strong></td>
                <td>${dept.description || 'No description'}</td>
                <td><span class="badge badge-primary">${dept.students ? dept.students.length : 0} students</span></td>
                <td class="actions">
                    <button class="btn btn-secondary btn-sm" onclick="app.editDepartment(${dept.id})">
                        Edit
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="app.deleteDepartment(${dept.id})">
                        Delete
                    </button>
                </td>
            </tr>
        `).join('');
    }

    renderStudents() {
        const tbody = document.getElementById('studentsTableBody');
        
        if (this.students.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty-state">
                        <h3>No students found</h3>
                        <p>Click "Add Student" to register your first student</p>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = this.students.map(student => {
            const dept = this.departments.find(d => d.id === student.department_id);
            return `
                <tr>
                    <td><strong>${student.first_name} ${student.last_name}</strong></td>
                    <td>${student.student_id}</td>
                    <td>${student.email}</td>
                    <td>${student.phone || 'N/A'}</td>
                    <td>${dept ? dept.name : 'No Department'}</td>
                    <td class="actions">
                        <button class="btn btn-secondary btn-sm" onclick="app.editStudent(${student.id})">
                            Edit
                        </button>
                        <button class="btn btn-danger btn-sm" onclick="app.deleteStudent(${student.id})">
                            Delete
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    updateDepartmentSelect() {
        const select = document.getElementById('studentDepartment');
        select.innerHTML = '<option value="">Select Department (Optional)</option>' +
            this.departments.map(dept => 
                `<option value="${dept.id}">${dept.name}</option>`
            ).join('');
    }

    showDepartmentModal(department = null) {
        const modal = document.getElementById('departmentModal');
        const form = document.getElementById('departmentForm');
        const title = document.getElementById('departmentModalTitle');
        
        if (department) {
            title.textContent = 'Edit Department';
            document.getElementById('departmentId').value = department.id;
            document.getElementById('departmentName').value = department.name;
            document.getElementById('departmentDescription').value = department.description || '';
        } else {
            title.textContent = 'Add Department';
            form.reset();
            document.getElementById('departmentId').value = '';
        }
        
        modal.classList.add('show');
    }

    showStudentModal(student = null) {
        const modal = document.getElementById('studentModal');
        const form = document.getElementById('studentForm');
        const title = document.getElementById('studentModalTitle');
        
        if (student) {
            title.textContent = 'Edit Student';
            document.getElementById('studentId').value = student.id;
            document.getElementById('studentFirstName').value = student.first_name;
            document.getElementById('studentLastName').value = student.last_name;
            document.getElementById('studentEmail').value = student.email;
            document.getElementById('studentStudentId').value = student.student_id;
            document.getElementById('studentPhone').value = student.phone || '';
            document.getElementById('studentAddress').value = student.address || '';
            document.getElementById('studentDepartment').value = student.department_id || '';
        } else {
            title.textContent = 'Add Student';
            form.reset();
            document.getElementById('studentId').value = '';
        }
        
        modal.classList.add('show');
    }

    closeModal() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.classList.remove('show');
        });
    }

    async saveDepartment() {
        const form = document.getElementById('departmentForm');
        const formData = new FormData(form);
        const departmentId = formData.get('id');
        
        const data = {
            name: formData.get('name'),
            description: formData.get('description')
        };

        try {
            let response;
            if (departmentId) {
                response = await fetch(`${this.apiBase}/departments/${departmentId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
            } else {
                response = await fetch(`${this.apiBase}/departments/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
            }

            if (response.ok) {
                this.showAlert(`Department ${departmentId ? 'updated' : 'created'} successfully!`, 'success');
                this.closeModal();
                this.loadDepartments();
            } else {
                const error = await response.json();
                this.showAlert(error.detail || 'Error saving department', 'error');
            }
        } catch (error) {
            this.showAlert('Error saving department: ' + error.message, 'error');
        }
    }

    async saveStudent() {
        const form = document.getElementById('studentForm');
        const formData = new FormData(form);
        const studentId = formData.get('id');
        
        const data = {
            first_name: formData.get('first_name'),
            last_name: formData.get('last_name'),
            email: formData.get('email'),
            student_id: formData.get('student_id'),
            phone: formData.get('phone'),
            address: formData.get('address'),
            department_id: formData.get('department_id') ? parseInt(formData.get('department_id')) : null
        };

        try {
            let response;
            if (studentId) {
                response = await fetch(`${this.apiBase}/students/${studentId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
            } else {
                response = await fetch(`${this.apiBase}/students/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
            }

            if (response.ok) {
                this.showAlert(`Student ${studentId ? 'updated' : 'created'} successfully!`, 'success');
                this.closeModal();
                this.loadStudents();
            } else {
                const error = await response.json();
                this.showAlert(error.detail || 'Error saving student', 'error');
            }
        } catch (error) {
            this.showAlert('Error saving student: ' + error.message, 'error');
        }
    }

    async editDepartment(id) {
        const department = this.departments.find(d => d.id === id);
        if (department) {
            this.showDepartmentModal(department);
        }
    }

    async editStudent(id) {
        const student = this.students.find(s => s.id === id);
        if (student) {
            this.showStudentModal(student);
        }
    }

    async deleteDepartment(id) {
        if (!confirm('Are you sure you want to delete this department?')) return;

        try {
            const response = await fetch(`${this.apiBase}/departments/${id}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showAlert('Department deleted successfully!', 'success');
                this.loadDepartments();
                this.loadStudents(); // Refresh students as well
            } else {
                const error = await response.json();
                this.showAlert(error.detail || 'Error deleting department', 'error');
            }
        } catch (error) {
            this.showAlert('Error deleting department: ' + error.message, 'error');
        }
    }

    async deleteStudent(id) {
        if (!confirm('Are you sure you want to delete this student?')) return;

        try {
            const response = await fetch(`${this.apiBase}/students/${id}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showAlert('Student deleted successfully!', 'success');
                this.loadStudents();
            } else {
                const error = await response.json();
                this.showAlert(error.detail || 'Error deleting student', 'error');
            }
        } catch (error) {
            this.showAlert('Error deleting student: ' + error.message, 'error');
        }
    }

    showAlert(message, type = 'success') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    showLoading(elementId) {
        document.getElementById(elementId).classList.add('show');
    }

    hideLoading(elementId) {
        document.getElementById(elementId).classList.remove('show');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SchoolManagementApp();
});