// run node with import support
// node generate.js --experimental-modules

import {QUERIES as qs} from "./queries.js";
import fs from 'fs';
import DiffMatchPatch from 'diff-match-patch';

let queries = []


function Query(q, id) {
    // read and parse q.state json file
    const dmp = new DiffMatchPatch()
    const resultAll = fs.readFileSync(q.state).toString()
    const resultJSON = []
    let prevString = ""
    for (let line of resultAll.split("\n").filter(e => e)) {
        const diffText = line
        const patches = JSON.parse(diffText)
        const res = dmp.patch_apply(patches, prevString)
        const patchedLine = res[0]
        prevString = patchedLine
        const jsonFromLine = JSON.parse(patchedLine)
        resultJSON.push(jsonFromLine)
    }

    const template = `
<div id="${id}" class="side-by-side compact" datafilepath="${q.state}">
    <div class="output output-height">
        <div class="chat">
          <span class="prompt tag-system">
          <span class="content" style="font-family: 'Google Sans', monospace;">
            You are chatting with an LLM (ChatGPT in this case). Click a sentence to see the effect of ChatProtect.
          </span>
          </span>
        <span class="prompt tag-user">
        <span class="content">
        ${q.prompt}
        </span>
        </span>
        <span class="variable v0 tag-assistant">
        <span class="content">
        ${resultJSON.slice(-1)[0].sentences.join(" ")}
        </span>
        </span>
        <span class="prompt tag-alternative"><span class="content"></span></span>
        </div>
        <div class="play-control">
          <a id="player-playpause">⏯️️</a>
          <a id="player-forward">⏩</a>
          <a id="player-stop">⏹️</a>
          <a id="player-end">⏭️️</a>
        </div>
        <div class="slidecontainer">
            <input type="range" min="0" max="${resultJSON.length-1}" value="0" class="slider" dataid="${id}">
        </div>
    </div>
</div>`
    return template
}

const prompts = {}
const showcase = JSON.parse(fs.readFileSync("showcase-queries.json"))
let q_text = ""
for (let q of qs) {
    if (showcase.includes(q.state)) {
        let id = q.state.replaceAll("/", "-").replaceAll(".", "-")
        q_text += Query(q, id)
        queries.push({
            "name": q.name || "Untitled",
            "id": id,
            "emoji": q.emoji || "❓️",
            "image": q.image || "2753",
        })
        prompts[q.prompt.trim()] = id
    } else {
        console.log("Skipping " + q.state)
    }
}

const index_html = fs.readFileSync("index.template.html")
let index_html_output = index_html.toString().replace("<%SAMPLES%>", q_text)
let query_options = queries.map((q,_) => `<span class="option example" value="${q.id}"><table><tr><td class="line-height"><img src="/static/images/emojis/${q.image}.svg" class="inline-emoji"></td><td>&nbsp;${q.name}</td></tr></table></span>`).join("\n")
index_html_output = index_html_output.replace("<%SAMPLES_LIST%>", query_options)

index_html_output = index_html_output.replace("<%PROMPTS%>", JSON.stringify(prompts))
// last update date and hour+minute Zurich time
const now = new Date()
const formatted = new Date().toLocaleString('en-US', { timeZone: 'Europe/Zurich', weekday: 'short', month: 'short', day: 'numeric' , hour: 'numeric', minute: 'numeric' })
const timezone = now.toLocaleString('en-US', { timeZoneName: 'short' }).split(' ').pop()
index_html_output = index_html_output.replace("<%LAST_UPDATED%>", formatted + " (" + timezone + ")")

fs.writeFileSync("index.html", index_html_output)