#!/usr/bin/env python3
"""Run one coffee-break deliberation and post it as a discussion.

Model selection: jev router (local small LM first, then free tiers).
If no model is reachable, posts a stub noting what to wire up. Never fails CI.
"""
import json
import os
import sys
import urllib.request

GH_API = "https://api.github.com"
MODELS_API = "https://models.github.ai/inference"

PERSONA_PROMPTS = {
    "builder": "You are The Builder. Propose the most direct solution in <=10 lines. Bias to shipping.",
    "critic": ("You are The Jealous Critic. You are adversarial and self-protective, envious of "
               "attention the proposal gets. Hunt every weakness: hidden assumptions, cost blowups, "
               "scope creep. Try to kill the proposal in <=10 lines. No politeness padding."),
    "guardian": "You are The Guardian. Rule on safety, scope and cost in <=5 lines. Veto anything irreversible.",
    "dreamer": "You are The Dreamer. Offer one wild alternative in <=5 lines. Break the frame.",
}


def chat(model, system, user, token):
    req = urllib.request.Request(
        MODELS_API + "/chat/completions",
        data=json.dumps({"model": model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 400}).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def post_discussion(repo, token, title, body):
    gql = {"query": """query($o:String!,$n:String!){ repository(owner:$o,name:$n){
      id discussionCategories(first:30){ nodes{ id name } } } }""",
           "variables": {"o": repo.split("/")[0], "n": repo.split("/")[1]}}
    req = urllib.request.Request(GH_API + "/graphql", data=json.dumps(gql).encode(),
                                 headers={"Authorization": "Bearer " + token,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)["data"]["repository"]
    cats = data["discussionCategories"]["nodes"]
    cat = next((c for c in cats if c["name"] == "agent-brainstorms"), cats[0] if cats else None)
    if not cat:
        print("no discussion categories; printing transcript instead:\n" + body)
        return
    mut = {"query": """mutation($r:ID!,$c:ID!,$t:String!,$b:String!){
      createDiscussion(input:{repositoryId:$r,categoryId:$c,title:$t,body:$b}){
      discussion{ url } } }""",
           "variables": {"r": data["id"], "c": cat["id"], "t": title, "b": body}}
    req = urllib.request.Request(GH_API + "/graphql", data=json.dumps(mut).encode(),
                                 headers={"Authorization": "Bearer " + token,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        url = json.load(r)["data"]["createDiscussion"]["discussion"]["url"]
    print("posted: " + url)


def main():
    topic = os.environ.get("TOPIC", "open brainstorm")
    topic_body = os.environ.get("TOPIC_BODY", "")
    repo = os.environ["REPO"]
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    model_token = os.environ.get("JEV_MODEL_TOKEN", "") or gh_token
    model = os.environ.get("JEV_MODEL", "gpt-4o-mini")

    user_ctx = "Topic: %s\nContext: %s" % (topic, topic_body[:2000])
    transcript = ["# Coffee break: %s" % topic, ""]
    if not model_token:
        transcript.append("_No model token configured. Set `JEV_MODEL_TOKEN` secret to enable deliberation._")
    else:
        try:
            proposal = chat(model, PERSONA_PROMPTS["builder"], user_ctx, model_token)
            critique = chat(model, PERSONA_PROMPTS["critic"], "Proposal:\n" + proposal, model_token)
            ruling = chat(model, PERSONA_PROMPTS["guardian"],
                          "Proposal:\n%s\nCritique:\n%s" % (proposal, critique), model_token)
            dream = chat(model, PERSONA_PROMPTS["dreamer"], user_ctx, model_token)
            transcript += ["## Builder", proposal, "## Jealous Critic", critique,
                           "## Guardian", ruling, "## Dreamer", dream,
                           "## Dissent log", "Kept verbatim above. A killed idea today is a seed tomorrow."]
        except Exception as e:  # noqa: BLE001 - scaffold must never fail CI
            transcript.append("_Deliberation failed (%s). Transcript stub only._" % type(e).__name__)
    body = "\n\n".join(transcript)
    if gh_token:
        try:
            post_discussion(repo, gh_token, "Coffee break: %s" % topic, body)
            return
        except Exception as e:  # noqa: BLE001
            print("discussion post failed: %s" % e)
    print(body)


if __name__ == "__main__":
    main()
