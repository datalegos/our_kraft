# Chatbot Test Questions - Hard & Indirect

## 🎯 **Hardest Questions (Require Deep Understanding)**

### Team & Leadership Questions
1. **"Who founded the company and what's their background?"**
   - Tests: Extracting founder info, background details, multiple pieces of information

2. **"Tell me about the people behind DataLegos"**
   - Tests: Indirect team question, comprehensive extraction

3. **"Who should I reach out to if I need help with graph database architecture?"**
   - Tests: Inference based on roles/expertise, connecting team to services

4. **"What experience does the leadership team have?"**
   - Tests: Aggregating information across multiple team members

5. **"Who are the key decision makers at DataLegos?"**
   - Tests: Understanding hierarchy, identifying leadership roles

6. **"Can you introduce me to the team members and their specialties?"**
   - Tests: Comprehensive team listing with roles

### Address & Location Questions
7. **"Where is DataLegos located?"**
   - Tests: Multiple addresses (corporate vs registered), complete address extraction

8. **"What's the mailing address for the company?"**
   - Tests: Distinguishing between different address types

9. **"Where can I send official correspondence?"**
   - Tests: Understanding registered office vs operational office

10. **"Are there multiple office locations?"**
    - Tests: Identifying and listing all addresses

### Contact Information Questions
11. **"How can I get in touch with someone?"**
    - Tests: Comprehensive contact info extraction (phone, email, hours)

12. **"What are your business hours?"**
    - Tests: Extracting specific timing information

13. **"Is there a phone number I can call?"**
    - Tests: Direct contact detail extraction

14. **"What's the best way to contact you during weekdays?"**
    - Tests: Combining hours and contact methods

### Services & Expertise Questions
15. **"What does DataLegos actually do?"**
    - Tests: Comprehensive service description, not just listing

16. **"What makes DataLegos different from other consulting companies?"**
    - Tests: Extracting unique value propositions, competitive advantages

17. **"What industries do you serve?"**
    - Tests: Listing all industries mentioned

18. **"Can you help me with fraud detection?"**
    - Tests: Service matching, understanding capabilities

19. **"What technologies does DataLegos specialize in?"**
    - Tests: Extracting technical expertise (Neo4j, AI, etc.)

20. **"Do you offer training programs?"**
    - Tests: Career Catalyst program, training services

### Company Information Questions
21. **"How long has DataLegos been in business?"**
    - Tests: Extracting "over a decade" or similar temporal information

22. **"What's the company's approach to client relationships?"**
    - Tests: Extracting philosophy, values, approach from team quotes

23. **"What kind of projects has DataLegos completed?"**
    - Tests: Extracting project examples, case studies, numbers (50+ deployments)

24. **"What's DataLegos' track record?"**
    - Tests: Metrics, success stories, client retention rates

25. **"How does DataLegos help startups?"**
    - Tests: Specific service offerings for startups

## 🔄 **Indirect Questions (Require Inference)**

### Indirect Team Questions
26. **"I'm looking for someone with graph database expertise. Who can help?"**
    - Tests: Role matching, expertise inference

27. **"Who would be the right person to discuss client solutions with?"**
    - Tests: Understanding roles and responsibilities

28. **"I need to speak with someone about operational matters."**
    - Tests: Role-based routing inference

29. **"Who mentors the Career Catalyst program?"**
    - Tests: Connecting team members to programs

### Indirect Contact Questions
30. **"I'm in India and want to reach out. What's the best time?"**
    - Tests: Timezone understanding, IST hours

31. **"How quickly can I expect a response if I contact you?"**
    - Tests: Business hours inference, response expectations

32. **"What's the preferred method of communication?"**
    - Tests: Multiple contact methods, preference inference

### Indirect Service Questions
33. **"I have a data problem. Can you help?"**
    - Tests: Service matching, understanding capabilities

34. **"We're struggling with data connections. Is this something you do?"**
    - Tests: Understanding "graph intelligence" and "connections"

35. **"Do you work with Neo4j?"**
    - Tests: Technology expertise extraction

36. **"We need help with knowledge graphs. Is that your specialty?"**
    - Tests: Core competency identification

37. **"Can you help us build an MVP quickly?"**
    - Tests: Startup services, MVP development

### Indirect Company Questions
38. **"What's your company culture like?"**
    - Tests: Extracting values from team quotes, approach descriptions

39. **"How do you approach client partnerships?"**
    - Tests: Philosophy extraction from leadership quotes

40. **"What's your success rate with clients?"**
    - Tests: Metrics extraction (95% retention, 50+ deployments)

## 🧩 **Complex Multi-Part Questions**

41. **"Tell me everything about DataLegos - team, services, location, and how to contact you"**
    - Tests: Comprehensive information extraction, multiple topics

42. **"Who are the founders, what do they do, and how can I reach them?"**
    - Tests: Multiple information types in one query

43. **"What services do you offer, who provides them, and where are you based?"**
    - Tests: Service-team-location connection

44. **"Give me a complete overview: leadership, team, expertise, and contact information"**
    - Tests: Full company profile extraction

## 🎭 **Ambiguous/Vague Questions**

45. **"Tell me about you"**
    - Tests: Understanding "you" refers to company, comprehensive response

46. **"What should I know?"**
    - Tests: Most important information extraction

47. **"Who are you?"**
    - Tests: Company introduction, not just name

48. **"Help me understand DataLegos"**
    - Tests: Comprehensive explanation

49. **"What's the story?"**
    - Tests: Company history, mission, approach

50. **"Give me the highlights"**
    - Tests: Key information prioritization

## 🔍 **Edge Cases & Tricky Questions**

51. **"Do you have an office in the USA?"**
    - Tests: Only India addresses exist, should clarify

52. **"What's the CEO's name?"**
    - Tests: No CEO mentioned, should identify founder/principal instead

53. **"How many employees do you have?"**
    - Tests: Exact number not in PDF, should list team members

54. **"What's your revenue?"**
    - Tests: Information not available, should handle gracefully

55. **"Can you do machine learning projects?"**
    - Tests: AI/ML expertise mentioned, should confirm

56. **"Do you offer remote services?"**
    - Tests: Hybrid work model mentioned, should infer capability

57. **"What certifications do your team members have?"**
    - Tests: Neo4j-certified mentioned, should extract

58. **"Have you worked with Fortune 500 companies?"**
    - Tests: Yes, mentioned in context, should confirm

59. **"What's your client retention rate?"**
    - Tests: 95% mentioned, should extract specific metric

60. **"How many projects have you completed?"**
    - Tests: 50+ deployments mentioned, should extract

## 📊 **Questions Requiring Data Aggregation**

61. **"List all the team members and their roles"**
    - Tests: Complete team listing with roles

62. **"What are all the services you offer?"**
    - Tests: Comprehensive service listing

63. **"What industries have you worked in?"**
    - Tests: All industries listed (Finance, Healthcare, Retail, Logistics, Startups)

64. **"What technologies are you experts in?"**
    - Tests: Neo4j, AI, ML, Python, etc.

65. **"What are all the ways I can contact you?"**
    - Tests: Phone, email, office hours

## 🎯 **Questions Testing Specific Details**

66. **"What's the exact postal code of your office?"**
    - Tests: 524002 extraction

67. **"What state is DataLegos in?"**
    - Tests: Andhra Pradesh extraction

68. **"What's the phone number format?"**
    - Tests: +91-8179301110 extraction

69. **"What's the email domain?"**
    - Tests: @data-legos.com extraction

70. **"What time does your office close?"**
    - Tests: 10:00 pm IST extraction

## 🧠 **Questions Requiring Inference**

71. **"If I'm a startup, what can DataLegos do for me?"**
    - Tests: Startup-specific services, MVP development

72. **"What's the best use case for your services?"**
    - Tests: Understanding primary use cases (fraud detection, supply chain, etc.)

73. **"Why should I choose DataLegos over competitors?"**
    - Tests: Unique value propositions, vendor-neutral approach

74. **"What's your biggest strength?"**
    - Tests: Core competencies, graph expertise, team experience

75. **"How do you ensure client success?"**
    - Tests: Approach, methodology, client retention strategies

---

## 📝 **Testing Checklist**

After testing, check if the chatbot:
- ✅ Extracts ALL relevant information (not just partial)
- ✅ Handles indirect questions correctly
- ✅ Provides complete addresses (not just city)
- ✅ Lists all team members (not just leadership)
- ✅ Includes all contact methods
- ✅ Handles questions about unavailable information gracefully
- ✅ Connects related information (e.g., team roles to services)
- ✅ Provides structured, well-formatted answers
- ✅ Doesn't say "I don't know" when information exists
- ✅ Handles ambiguous questions appropriately

## 🎯 **Priority Test Questions (Start Here)**

If you want to test quickly, start with these 10:

1. "Tell me about the team" (comprehensive)
2. "Where is DataLegos located?" (multiple addresses)
3. "How can I contact you?" (all contact info)
4. "What does DataLegos do?" (services)
5. "Who founded the company?" (founder details)
6. "What's your business hours?" (specific timing)
7. "What industries do you serve?" (list all)
8. "Who should I contact for graph database help?" (inference)
9. "Give me a complete overview of DataLegos" (comprehensive)
10. "What makes DataLegos different?" (value proposition)

Good luck with testing! 🚀

