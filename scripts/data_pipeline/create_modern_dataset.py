import json

modern_pairs = [
    {
        "prompt": "write me a mail asking my boss for a leave tomorrow",
        "chosen": "Hi [Boss], I need to take tomorrow off for a personal matter. I'll make sure my team has my responsibilities covered. Thanks.",
        "rejected": "Subject: Request for Leave Tomorrow\nDear [Boss's Name],\nI hope this message finds you well. I am writing to request a leave for tomorrow. I have an important personal matter that requires my attention and cannot be postponed. I understand that this may affect our work, but I assure you that I will ensure that my responsibilities are covered by my team. I will be available to answer any urgent emails and messages before and after my absence. Please let me know if this is acceptable. Thank you for considering my request. I appreciate your understanding.\nBest regards,\n[Your Name]"
    },
    {
        "prompt": "Draft an email to Mark asking for the latest TPS reports",
        "chosen": "Mark, can you send over the latest TPS reports when you get a chance? Thanks.",
        "rejected": "Subject: Request for Latest TPS Reports\nHi Mark,\nI hope you are having a productive week. I was wondering if you could provide me with the latest TPS (Transaction Per Second) reports. I need this information for a presentation I'm preparing for the upcoming meeting. Your prompt assistance in this matter would be greatly appreciated. Please let me know if you need anything else from my end.\nThanks,\n[Your Name]"
    },
    {
        "prompt": "Write a quick update to the engineering team. Let them know the database migration is delayed until next Tuesday because of some unexpected downtime with the AWS servers. Tell them to keep working on the frontend tickets in the meantime and ping me if they are blocked.",
        "chosen": "Team, the database migration is delayed to next Tuesday due to AWS downtime. Please continue working on your frontend tickets and ping me if you have any blockers.",
        "rejected": "Subject: Delayed Database Migration Update\nHi Engineering Team,\nI hope everyone is having a good day. The database migration is now delayed until next Tuesday due to some unexpected downtime with the AWS servers. This means you will need to continue working on the frontend tickets until then. Please keep up with the frontend work and let me know if you encounter any blockers or need assistance. We appreciate your flexibility as we work through these technical difficulties.\nThanks,\n[Your Name]"
    },
    {
        "prompt": "Draft an email to a client named Sarah. Apologize for missing our scheduled 10 AM zoom call. Explain that my previous meeting ran 30 minutes over and I lost track of time. Ask if we can reschedule for tomorrow at the same time.",
        "chosen": "Hi Sarah, I'm so sorry I missed our 10 AM Zoom call. My previous meeting ran 30 minutes over and I completely lost track of time. Can we reschedule for tomorrow at 10 AM?",
        "rejected": "Subject: Rescheduling of 10 AM Zoom Call\nHi Sarah,\nI hope you're doing well. I wanted to reach out to sincerely apologize for missing our scheduled 10 AM Zoom call today. My previous meeting ran 30 minutes over and unfortunately I lost track of time. I know your time is valuable and I apologize for the inconvenience. Would you be open to rescheduling our call for tomorrow at the same time? I appreciate your understanding and flexibility. If you have any preferences for the time or any other adjustments, please let me know.\nBest, [Your Name]"
    },
    {
        "prompt": "Write an email to the marketing group kicking off the new Q3 ad campaign. Tell them I've attached the initial brief and we'll have a brainstorming session on Thursday. Tell them to read the brief before the meeting. Sign it from Alex.",
        "chosen": "Team, attached is the initial brief for the Q3 ad campaign. Please review it before our brainstorming session this Thursday. Thanks,\nAlex",
        "rejected": "Subject: Kickoff of Q3 Ad Campaign - Initial Brief and Brainstorming Meeting\nHi Marketing Team,\nThanks for your continued support and enthusiasm for our upcoming Q3 ad campaign. We're excited to kick things off with a bang and I'm looking forward to our brainstorming session to refine our approach and set the tone for success.\nHere's what I've prepared:\n1. Initial Brief: This document outlines our key objectives, target audience, and creative direction for the campaign.\n2. Brainstorming Session: We'll be meeting on Thursday to dive into these details and come up with ideas that align with the brief.\nPlease take a look at the initial brief and prepare your thoughts on our target audience, campaign goals, and any specific elements you'd like to include. We want to ensure that our campaign is not only effective but also resonates with our audience. The meeting will be a great opportunity to discuss our ideas and refine them together. If you have any questions or need any additional information, feel free to reach out.\nSee you on Thursday.\nBest,\nAlex"
    },
    {
        "prompt": "Send a quick note to Mike. Ask him if he's still free to grab lunch today at the sandwich shop across the street. Mention that I need to pick his brain about the new hiring budget.",
        "chosen": "Mike, still free for lunch today at the sandwich shop? Need to pick your brain about the new hiring budget.",
        "rejected": "Subject: Lunch with Mike - Need to Brainstorm Hiring Budget\nHey Mike,\nI hope your morning is going well. I was wondering if you're still available to grab lunch today at the sandwich shop across the street. I need to get your opinion on the new hiring budget. I think it's a critical issue for our team, and I want to get your insights before we finalize the numbers. Let me know if that still works for you.\nThanks!"
    },
    {
        "prompt": "Write an email to my manager, David. Tell him I don't think we have the bandwidth to take on the 'Project Phoenix' deliverables this sprint unless we drop the bug fixes we planned. Ask him which one he wants us to prioritize.",
        "chosen": "David, we don't have bandwidth for the Project Phoenix deliverables this sprint unless we drop the planned bug fixes. Which would you prefer we prioritize?",
        "rejected": "Subject: Priority Review for Project Phoenix Deliverables\nHi David,\nI wanted to discuss our sprint planning for the Project Phoenix deliverables. We have identified two main tasks: developing the core features and fixing bugs. However, given the current workload and our resource constraints, I believe we need to reassess the priorities. For the core development, we have a set of requirements that need to be implemented. This is a critical task that will lead to the successful completion of the Project Phoenix. However, the bug fixes are also crucial, as they ensure the stability and reliability of the application. Without addressing these bugs, we risk delaying the project and impacting the end-users. I propose we focus on completing the core development tasks this sprint. Once we have the code ready, we can then address the bugs. This way, we can deliver the most critical features first, and the bug fixes can be done in subsequent sprints. Would you like to proceed with this plan? If so, please let me know, and I'll coordinate with the team to ensure we're on track.\nThanks,\n[Your Name]"
    },
    {
        "prompt": "Ask IT to reset my VPN password. My username is jsmith.",
        "chosen": "Hi IT, could you please reset the VPN password for my account (jsmith)? Thanks.",
        "rejected": "Subject: Request for VPN Password Reset\nDear IT Department,\nI hope this email finds you well. I am writing to formally request a password reset for my VPN account. My username is jsmith. I have been unable to log in and need access to the network for my daily tasks. Please let me know what steps I need to take to verify my identity or if there is a temporary password you can provide. I appreciate your prompt attention to this matter as it is impacting my ability to work remotely.\nThank you,\nJohn Smith"
    },
    {
        "prompt": "Email Jane to see if she reviewed the Q2 financials spreadsheet I sent yesterday.",
        "chosen": "Hi Jane, did you get a chance to review the Q2 financials spreadsheet I sent yesterday?",
        "rejected": "Subject: Following up on Q2 Financials Spreadsheet\nHi Jane,\nI hope your week is off to a great start. I am writing to follow up on the Q2 financials spreadsheet that I sent over to you yesterday afternoon. I just wanted to check in and see if you have had a chance to review it yet. Your feedback is crucial before we present the numbers to the executive team. Please take your time, but if you could let me know when you might have some notes for me, I would greatly appreciate it.\nBest regards,\n[Your Name]"
    },
    {
        "prompt": "Tell the team that the weekly sync is moved to 2 PM today because I have a doctor's appointment.",
        "chosen": "Team, I have a doctor's appointment today, so I'm moving our weekly sync to 2 PM.",
        "rejected": "Subject: Rescheduling Weekly Sync to 2 PM\nHi Team,\nI hope everyone is having a productive day. I am writing to inform you that we will need to reschedule our weekly sync meeting today. Unfortunately, I have a doctor's appointment that conflicts with our original time slot. Therefore, we will be moving the meeting to 2 PM today instead. I apologize for any inconvenience this may cause and appreciate your flexibility. Please make sure to update your calendars and come prepared to discuss our usual agenda items at 2 PM. Let me know if this new time causes major conflicts for anyone.\nThanks,\n[Your Name]"
    },
    {
        "prompt": "Send a Jira ticket to DevOps to spin up a new staging environment.",
        "chosen": "Hi DevOps, could you please spin up a new staging environment for the upcoming release? I've created Jira ticket DEV-123 for this.",
        "rejected": "Subject: Request for New Staging Environment\nDear DevOps Team,\nI hope you are having a wonderful week so far. I am reaching out to request that a new staging environment be spun up for our upcoming release cycle. Having a dedicated staging environment is crucial for us to perform comprehensive testing without disrupting current operations. I have gone ahead and created a Jira ticket for this request (DEV-123) with more detailed specifications. Could you please take a look at it and let me know the estimated timeline for completion? We appreciate all the hard work you do to support our development efforts.\nBest,\n[Your Name]"
    },
    {
        "prompt": "Ask the design team when the Figma mocks for the landing page will be ready.",
        "chosen": "Hi Design Team, do we have an ETA on when the Figma mocks for the new landing page will be ready?",
        "rejected": "Subject: Inquiry Regarding Figma Mocks Timeline\nHello Design Team,\nI hope this email finds you well. I am writing to check in on the status of the Figma mocks for the new landing page project. We are eager to begin the development phase and need these designs to proceed. Could you please provide an estimated time of arrival for when these might be ready for our review? We completely understand you have multiple priorities, but a rough timeline would help us schedule our upcoming sprint tasks accordingly. Thank you for your continued collaboration and excellent work.\nBest regards,\n[Your Name]"
    },
    {
        "prompt": "Tell the client we received their feedback and will have a revised draft by Friday.",
        "chosen": "Thanks for the feedback! We will send you a revised draft by Friday.",
        "rejected": "Subject: Confirmation of Feedback Receipt and Next Steps\nDear Client,\nThank you so much for taking the time to provide us with your valuable feedback. We have successfully received your comments and our team is currently reviewing them in detail. We greatly appreciate your input, as it helps us ensure the final product aligns perfectly with your vision. We will incorporate your suggestions and work diligently to produce a revised draft. You can expect to receive the updated version from us by this coming Friday. Please feel free to reach out if you have any additional thoughts in the meantime.\nWarmly,\n[Your Name]"
    },
    {
        "prompt": "Email the office manager asking if we can get oat milk stocked in the 3rd floor fridge.",
        "chosen": "Hi, could we start stocking oat milk in the 3rd floor fridge? Thanks!",
        "rejected": "Subject: Request for Oat Milk in 3rd Floor Refrigerator\nHello Office Manager,\nI hope you are having a pleasant day. I am writing to submit a request regarding the pantry supplies on the 3rd floor. A few of us have noticed that we currently only have regular milk available, and we were wondering if it might be possible to start stocking oat milk in the 3rd floor fridge as a dairy-free alternative? It would be greatly appreciated by several team members who prefer or require dairy alternatives. Thank you so much for considering this request and for all you do to keep our office running smoothly.\nBest,\n[Your Name]"
    },
    {
        "prompt": "Remind everyone to submit their timesheets before 5 PM today.",
        "chosen": "Hi everyone, friendly reminder to submit your timesheets before 5 PM today. Thanks!",
        "rejected": "Subject: Friendly Reminder: Timesheet Submission Deadline\nDear Team,\nI hope you've all had a productive week. As we approach the end of the day, I wanted to send a quick, friendly reminder regarding our timesheet submissions. Please ensure that you have fully completed and submitted your timesheets before 5 PM today. Timely submission is crucial for our payroll processing and project tracking. If you encounter any issues or have questions regarding the timesheet system, please do not hesitate to reach out to me or the HR department. Thank you in advance for your cooperation and prompt attention to this matter.\nBest regards,\n[Your Name]"
    }
]

with open("dpo_data/modern_anti_slop.jsonl", "w") as f:
    for pair in modern_pairs:
        f.write(json.dumps(pair) + "\n")

combined = []
with open("dpo_data/dpo_pairs.jsonl", "r") as f:
    for line in f:
        if line.strip():
            combined.append(json.loads(line))

for pair in modern_pairs:
    combined.append(pair)

with open("dpo_data/dpo_combined.jsonl", "w") as f:
    for pair in combined:
        f.write(json.dumps(pair) + "\n")

print(f"Generated {len(modern_pairs)} modern pairs.")
print(f"Combined dataset has {len(combined)} pairs.")
