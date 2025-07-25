from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
from scraper import EenaduScraper
from config import CATEGORIES, NEWS_SOURCES

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Ensure proper UTF-8 encoding for Telugu text
    app.config['JSON_AS_ASCII'] = False
    
    @app.route("/")
    def get_ui():
        """Serve the UI"""
        return send_from_directory("static", "index.html")

    @app.route("/static/<path:filename>")
    def static_files(filename):
        """Serve static files"""
        return send_from_directory("static", filename)

    @app.route("/scrape", methods=["POST"])
    def scrape_news():
        """Scrape news articles from Eenadu"""
        start_time = time.time()
        
        try:
            data = request.get_json()
            if not data:
                data = {}
                
            source_name = data.get("source_name", "eenadu")
            category = data.get("category")
            limit = int(data.get("limit", 10))
            full_content = data.get("full_content", True)
            
            # Validate inputs
            if limit > 50:
                limit = 50
            elif limit < 1:
                limit = 5
            
            scraper = EenaduScraper()
            articles, errors = scraper.scrape_articles(
                category=category,
                limit=limit,
                full_content=full_content
            )
            
            duration = time.time() - start_time
            
            # Determine status
            if articles and not errors:
                status = "success"
            elif articles and errors:
                status = "partial_success"
            else:
                status = "failed"
            
            response = {
                "status": status,
                "source": source_name,
                "category": category,
                "total_articles": len(articles),
                "articles": articles,
                "errors": errors,
                "duration_seconds": round(duration, 2)
            }
            
            return jsonify(response)
            
        except Exception as e:
            duration = time.time() - start_time
            error_response = {
                "status": "failed",
                "source": data.get("source_name", "eenadu") if 'data' in locals() else "eenadu",
                "category": data.get("category") if 'data' in locals() else None,
                "total_articles": 0,
                "articles": [],
                "errors": [f"Server error: {str(e)}"],
                "duration_seconds": round(duration, 2)
            }
            return jsonify(error_response), 500

    @app.route("/categories", methods=["GET"])
    def get_categories():
        """Get available categories"""
        return jsonify({"categories": CATEGORIES})

    @app.route("/sources", methods=["GET"])
    def get_sources():
        """Get available news sources"""
        return jsonify({"sources": NEWS_SOURCES})

    @app.route("/api/info", methods=["GET"])
    def api_info():
        """API information for Postman testing"""
        return jsonify({
            "service": "GlobalPulse News Scraper API",
            "version": "2.0.0",
            "description": "Advanced Telugu News Intelligence Platform with 50+ sources and 200+ categories",
            "base_url": request.host_url,
            "endpoints": {
                "scrape": {
                    "method": "POST",
                    "url": "/scrape",
                    "description": "Scrape news articles from Telugu sources",
                    "content_type": "application/json",
                    "parameters": {
                        "source_name": {
                            "type": "string",
                            "required": False,
                            "default": "eenadu",
                            "description": "News source identifier"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "default": null,
                            "description": "Category filter (empty for homepage)"
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "default": 10,
                            "min": 1,
                            "max": 50,
                            "description": "Number of articles to scrape"
                        },
                        "full_content": {
                            "type": "boolean",
                            "required": False,
                            "default": True,
                            "description": "Extract full article content"
                        }
                    },
                    "example_request": {
                        "source_name": "eenadu",
                        "category": "politics",
                        "limit": 10,
                        "full_content": True
                    },
                    "response_format": {
                        "status": "success|partial_success|failed",
                        "source": "string",
                        "category": "string|null",
                        "total_articles": "integer",
                        "articles": "array",
                        "errors": "array",
                        "duration_seconds": "float"
                    }
                },
                "categories": {
                    "method": "GET",
                    "url": "/categories",
                    "description": "Get all available news categories",
                    "response_format": {
                        "categories": [
                            {
                                "value": "string",
                                "label": "string",
                                "group": "string"
                            }
                        ]
                    }
                },
                "sources": {
                    "method": "GET",
                    "url": "/sources",
                    "description": "Get all available news sources",
                    "response_format": {
                        "sources": [
                            {
                                "value": "string",
                                "label": "string",
                                "url": "string",
                                "type": "string"
                            }
                        ]
                    }
                },
                "health": {
                    "method": "GET",
                    "url": "/health",
                    "description": "Health check endpoint"
                }
            },
            "postman_collection": {
                "info": {
                    "name": "GlobalPulse News Scraper API",
                    "description": "Collection for testing Telugu news scraping API",
                    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
                },
                "item": [
                    {
                        "name": "Get API Info",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/api/info",
                                "host": ["{{base_url}}"],
                                "path": ["api", "info"]
                            }
                        }
                    },
                    {
                        "name": "Get Categories",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/categories",
                                "host": ["{{base_url}}"],
                                "path": ["categories"]
                            }
                        }
                    },
                    {
                        "name": "Get Sources",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/sources",
                                "host": ["{{base_url}}"],
                                "path": ["sources"]
                            }
                        }
                    },
                    {
                        "name": "Scrape News - Basic",
                        "request": {
                            "method": "POST",
                            "header": [
                                {
                                    "key": "Content-Type",
                                    "value": "application/json"
                                }
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": "{\n  \"source_name\": \"eenadu\",\n  \"limit\": 5,\n  \"full_content\": true\n}"
                            },
                            "url": {
                                "raw": "{{base_url}}/scrape",
                                "host": ["{{base_url}}"],
                                "path": ["scrape"]
                            }
                        }
                    },
                    {
                        "name": "Scrape News - With Category",
                        "request": {
                            "method": "POST",
                            "header": [
                                {
                                    "key": "Content-Type",
                                    "value": "application/json"
                                }
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": "{\n  \"source_name\": \"sakshi\",\n  \"category\": \"politics\",\n  \"limit\": 10,\n  \"full_content\": true\n}"
                            },
                            "url": {
                                "raw": "{{base_url}}/scrape",
                                "host": ["{{base_url}}"],
                                "path": ["scrape"]
                            }
                        }
                    }
                ]
            },
            "usage_examples": {
                "curl": {
                    "basic_scrape": "curl -X POST {{base_url}}/scrape -H \"Content-Type: application/json\" -d '{\"source_name\": \"eenadu\", \"limit\": 5}'",
                    "category_scrape": "curl -X POST {{base_url}}/scrape -H \"Content-Type: application/json\" -d '{\"source_name\": \"sakshi\", \"category\": \"politics\", \"limit\": 10}'",
                    "get_sources": "curl -X GET {{base_url}}/sources",
                    "get_categories": "curl -X GET {{base_url}}/categories"
                },
                "python": {
                    "basic_scrape": "import requests\n\nresponse = requests.post('{{base_url}}/scrape', json={'source_name': 'eenadu', 'limit': 5})\ndata = response.json()",
                    "category_scrape": "import requests\n\nresponse = requests.post('{{base_url}}/scrape', json={'source_name': 'sakshi', 'category': 'politics', 'limit': 10})\ndata = response.json()"
                },
                "javascript": {
                    "basic_scrape": "fetch('{{base_url}}/scrape', {\n  method: 'POST',\n  headers: {'Content-Type': 'application/json'},\n  body: JSON.stringify({source_name: 'eenadu', limit: 5})\n}).then(r => r.json()).then(data => console.log(data));",
                    "category_scrape": "fetch('{{base_url}}/scrape', {\n  method: 'POST',\n  headers: {'Content-Type': 'application/json'},\n  body: JSON.stringify({source_name: 'sakshi', category: 'politics', limit: 10})\n}).then(r => r.json()).then(data => console.log(data));"
                }
            }
        })

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "service": "GlobalPulse News Scraper"})

    @app.route("/postman-collection", methods=["GET"])
    def postman_collection():
        """Export Postman collection for easy API testing"""
        base_url = request.host_url.rstrip('/')
        
        collection = {
            "info": {
                "name": "GlobalPulse News Scraper API",
                "description": "Complete API collection for Telugu news scraping with 50+ sources and 200+ categories",
                "version": "2.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": base_url,
                    "type": "string"
                }
            ],
            "item": [
                {
                    "name": "API Information",
                    "item": [
                        {
                            "name": "Get API Info",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/api/info",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "info"]
                                },
                                "description": "Get comprehensive API documentation and usage examples"
                            }
                        },
                        {
                            "name": "Health Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/health",
                                    "host": ["{{base_url}}"],
                                    "path": ["health"]
                                },
                                "description": "Check API health status"
                            }
                        }
                    ]
                },
                {
                    "name": "Configuration",
                    "item": [
                        {
                            "name": "Get All Categories",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/categories",
                                    "host": ["{{base_url}}"],
                                    "path": ["categories"]
                                },
                                "description": "Retrieve all 200+ available news categories grouped by type"
                            }
                        },
                        {
                            "name": "Get All Sources",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/sources",
                                    "host": ["{{base_url}}"],
                                    "path": ["sources"]
                                },
                                "description": "Retrieve all 50+ Telugu news sources with metadata"
                            }
                        }
                    ]
                },
                {
                    "name": "News Scraping",
                    "item": [
                        {
                            "name": "Basic Scrape - Homepage",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"eenadu\",\n  \"limit\": 5,\n  \"full_content\": true\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape 5 articles from Eenadu homepage with full content"
                            }
                        },
                        {
                            "name": "Category Scrape - Politics",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"sakshi\",\n  \"category\": \"politics\",\n  \"limit\": 10,\n  \"full_content\": true\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape 10 politics articles from Sakshi with full content"
                            }
                        },
                        {
                            "name": "Sports Scrape - Cricket",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"tv9telugu\",\n  \"category\": \"cricket\",\n  \"limit\": 15,\n  \"full_content\": false\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape 15 cricket articles from TV9 Telugu with summaries only"
                            }
                        },
                        {
                            "name": "Entertainment Scrape - Tollywood",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"greatandhra\",\n  \"category\": \"tollywood\",\n  \"limit\": 20,\n  \"full_content\": true\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape 20 Tollywood articles from Great Andhra with full content"
                            }
                        },
                        {
                            "name": "Technology Scrape",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"gadgets360\",\n  \"category\": \"technology\",\n  \"limit\": 12,\n  \"full_content\": true\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape 12 technology articles from Gadgets 360 Telugu"
                            }
                        },
                        {
                            "name": "Maximum Articles Scrape",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"andhrajyothy\",\n  \"category\": \"telangana\",\n  \"limit\": 50,\n  \"full_content\": false\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Scrape maximum 50 Telangana articles from Andhra Jyothy with summaries"
                            }
                        }
                    ]
                },
                {
                    "name": "Error Testing",
                    "item": [
                        {
                            "name": "Invalid Source",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"invalid_source\",\n  \"limit\": 5\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Test error handling with invalid source"
                            }
                        },
                        {
                            "name": "Limit Exceeded",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\n  \"source_name\": \"eenadu\",\n  \"limit\": 100\n}"
                                },
                                "url": {
                                    "raw": "{{base_url}}/scrape",
                                    "host": ["{{base_url}}"],
                                    "path": ["scrape"]
                                },
                                "description": "Test limit validation (should cap at 50)"
                            }
                        }
                    ]
                }
            ]
        }
        
        response = jsonify(collection)
        response.headers['Content-Disposition'] = 'attachment; filename=GlobalPulse-API-Collection.json'
        return response

    @app.route("/swagger.json", methods=["GET"])
    def swagger_spec():
        """OpenAPI/Swagger specification for API documentation"""
        base_url = request.host_url.rstrip('/')
        
        swagger = {
            "openapi": "3.0.0",
            "info": {
                "title": "GlobalPulse News Scraper API",
                "description": "Advanced Telugu News Intelligence Platform with comprehensive coverage from 50+ reliable sources across 200+ categories",
                "version": "2.0.0",
                "contact": {
                    "name": "GlobalPulse API Support"
                }
            },
            "servers": [
                {
                    "url": base_url,
                    "description": "Production server"
                }
            ],
            "paths": {
                "/scrape": {
                    "post": {
                        "summary": "Scrape news articles",
                        "description": "Extract news articles from specified Telugu news source with optional category filtering",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "source_name": {
                                                "type": "string",
                                                "description": "News source identifier",
                                                "default": "eenadu"
                                            },
                                            "category": {
                                                "type": "string",
                                                "description": "Category filter (null for homepage)",
                                                "nullable": True
                                            },
                                            "limit": {
                                                "type": "integer",
                                                "description": "Number of articles to scrape",
                                                "minimum": 1,
                                                "maximum": 50,
                                                "default": 10
                                            },
                                            "full_content": {
                                                "type": "boolean",
                                                "description": "Extract full article content",
                                                "default": True
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful scraping operation",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["success", "partial_success", "failed"]
                                                },
                                                "source": {"type": "string"},
                                                "category": {"type": "string", "nullable": True},
                                                "total_articles": {"type": "integer"},
                                                "articles": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "title": {"type": "string"},
                                                            "summary": {"type": "string"},
                                                            "content": {"type": "string"},
                                                            "url": {"type": "string"},
                                                            "author": {"type": "string"},
                                                            "published_at": {"type": "string"}
                                                        }
                                                    }
                                                },
                                                "errors": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                },
                                                "duration_seconds": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/categories": {
                    "get": {
                        "summary": "Get available categories",
                        "description": "Retrieve all 200+ news categories grouped by type",
                        "responses": {
                            "200": {
                                "description": "List of categories",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "categories": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "value": {"type": "string"},
                                                            "label": {"type": "string"},
                                                            "group": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/sources": {
                    "get": {
                        "summary": "Get available sources",
                        "description": "Retrieve all 50+ Telugu news sources with metadata",
                        "responses": {
                            "200": {
                                "description": "List of news sources",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "sources": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "value": {"type": "string"},
                                                            "label": {"type": "string"},
                                                            "url": {"type": "string"},
                                                            "type": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "description": "Check API health status",
                        "responses": {
                            "200": {
                                "description": "API is healthy",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string"},
                                                "service": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return jsonify(swagger)
    
    return app