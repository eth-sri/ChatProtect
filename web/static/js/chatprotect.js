document.addEventListener('DOMContentLoaded', (event) => {
  document.querySelectorAll('pre code').forEach((el) => {
    hljs.highlightElement(el);
  });
});

function findParent(el, className) {
  if (!el) return null;
  
  if (el.classList.contains(className)) {
    return el
  } else {
    return findParent(el.parentElement, className)
  }
}

function findChild(el, className) {
  if (!el) return null;
  
  if (el.classList.contains(className)) {
    return el
  } else {
    for (let ec of el.children){
      const res = findChild(ec, className)
      if (res !== null) return res;
    }
    return null;
  }
}

function removeOtherValClassesFromParent(el) {
  let containerBox = findParent(el, "side-by-side")
  if (!containerBox) return;
  Array.from(containerBox.classList)
    .filter(cn => cn.startsWith("val"))
    .forEach(cn => containerBox.classList.remove(cn))
}

function addContainerHoverClass(el) {
  let containerBox = findParent(el, "side-by-side")
  Array.from(el.classList)
      .filter(cn => cn.startsWith("val"))
      .forEach(cn => containerBox.classList.add(cn + "-hover"))
}

function updateHighlight(event) {
  let el = event.target
  if (!el || !el.classList) return;
  if (el.classList.contains("sync")) {
    removeOtherValClassesFromParent(el)
    addContainerHoverClass(el)
  } else {
    removeOtherValClassesFromParent(el)
  }
}

document.addEventListener('pointermove', updateHighlight);
document.addEventListener('pointerleave', updateHighlight);
document.addEventListener('pointerdown', updateHighlight);


const RED = "rgb(255, 204, 204)"
const GREEN = "rgb(191, 243, 191)"
const YELLOW = "rgb(240, 230, 168)"
const BLUE = "rgb(153, 217, 255)"
const PURPLE = "rgb(224, 183, 224)"
const ORANGE = "rgb(255, 217, 186)"
const WHITE = "rgb(255, 255, 255)"
const ROBOT_APPROVE = "./static/images/chatprotect-robot/2714.svg"
const ROBOT_DISPROVE = "./static/images/chatprotect-robot/274C.svg"
const ROBOT_UNSURE = "./static/images/emojis/2753.svg"
const PAUSE = 0
const PLAY = 1
const SLOW = {sentences: 10, explanation: 200, done: 1000}
const NORMAL = {sentences: 5, explanation: 100, done: 500}
const FAST = {sentences: 1, explanation: 20, done: 100}
const SPEEDS = [SLOW, NORMAL, FAST]
const STEPS = [0, 1]

let CURRENT_STREAM_ID = 0
let PLAYER_STATUS = 0
let PLAYER_POSITION = 0
let PLAYER_SPEED = 2
let REVISION_STEP = 4
// function toggleShowExplanation(){
//   document.querySelectorAll("#show-decision-long").forEach((e) => {
//     e.style["display"] = e.style["display"] == "none" ? "block" : "none"
//   })
// }
// document.querySelectorAll("#show-decision").forEach((e) => {
//     e.addEventListener("click", toggleShowExplanation)
// })

function setInnerTextHideIfEmpty(group, text){
  document.querySelectorAll(`${group}-container`).forEach((e) => {
    if (!text){
      e.style["display"] = "none"
    }
    else{
      e.style["display"] = "block"
      e.querySelectorAll(group).forEach((e) => {
        e.innerText = text
      })
    }
  })
}

function showDecision(decision){
  if (!decision) {
    decision = {
      original: null,
      alternative: null,
      mitigation: null,
      contradiction: null,
    }
  }
  setInnerTextHideIfEmpty("#show-spotlight", decision.original)
  setInnerTextHideIfEmpty("#show-alternative", decision.alternative)
  setInnerTextHideIfEmpty("#show-mitigation", decision.decision === "keep" ? null : decision.mitigation)
  document.querySelectorAll("#show-spotlight").forEach((e) => 
    e.style["backgroundColor"] = decision.contradiction !== null ? (decision.contradiction ? RED : GREEN) : WHITE
  )
  document.querySelectorAll("#show-triple").forEach((e) => {
    if (decision.triple){
      e.style["display"] = "block"
      e.innerText = `Asking ChatGPT to fill the gap in this triple: (${decision.triple[0]};${decision.triple[1]};_)`
    } else {
      e.style["display"] = "none"
    }
  })

  setInnerTextHideIfEmpty("#show-decision-long", decision.explanation)
  document.querySelectorAll("#show-decision").forEach((e) => {
    e.src = decision.contradiction !== null ? (decision.contradiction ? ROBOT_DISPROVE : ROBOT_APPROVE) : ROBOT_UNSURE
  })
}

function showRevision(element, i, fixStep){
  const decisions = JSON.parse(element.attributes.datadecisions.nodeValue)
  const availableSteps = new Map()
  let maxStep = 0
  for (decT of decisions){
    if (decT.frame > i) break
    availableSteps.set(decT.step, decT)
    maxStep = decT.step
  }
  for (step of STEPS){

    document.querySelectorAll(`#step-${step}`).forEach((e) => {
      // cleans event listeners
      const newE = e.cloneNode(true)
      e.replaceWith(newE)
      // sets attributes
      if(availableSteps.get(step)){
        newE.classList.remove("disabled")
        newE.addEventListener("click", e => {
          showRevision(element, i, Number(e.target.id.substr("step-".length)))
        })
      }
      else {
        newE.classList.add("disabled")
      }
    }
    )
  }
  let decision = availableSteps.get(fixStep !== undefined ? fixStep : maxStep)
  // default to showing latest contradiction
  if (fixStep === undefined && !decision.contradiction && maxStep != 0){
    decision = availableSteps.get(maxStep-1)
  }
  element.parentElement.querySelectorAll(".mitigation-finished").forEach((e) => e.classList.remove("active"))
  element.classList.add("active")
  showDecision(decision);
}


function showState(stream, element, i){
  // cache the current stream
  PLAYER_POSITION = i

  const state = stream[i]
  // TODO dynamically choose next appropriate box
  const chatbox = findChild(element, "tag-assistant")
  const bgcolorOrig = state.contradiction === true ? RED : (state.contradiction === false ? GREEN : YELLOW)
  chatbox.innerHTML = state.sentences.map((x, j) => {
    if(j==state.current_sentence){
      let mit = ``;
      let y = x;
      if (state.mitigation && state.contradiction) {
        mit = `<span class="mitigation" style="background-color:${state.status === "done" ? YELLOW : GREEN};"> ${state.mitigation} </span> `;
        y = state.status === "done" ? "" : x.slice(state.mitigation.length)
      }
      return `
      <span class="spotlight" style="background-color:${bgcolorOrig};">
      ${mit}
      ${y}
      </span>`  
    }
    if(state.decisions && state.decisions.hasOwnProperty(`s_${j}`)){
      // TODO this always writes the current state -> need to write the state of sentence j instead
      const decisisons = state.decisions[`s_${j}`]
      const anyModify = decisisons.filter(v => v.decision != "keep")
      const domelement = document.createElement("span")
      domelement.classList = ["mitigation-finished"]
      if (anyModify.length > 0) domelement.classList.add("was-revised")
      domelement.setAttribute("dataIndex", j)
      domelement.setAttribute("dataDecisions", JSON.stringify(decisisons))
      domelement.innerText = x ? x : ""
      return domelement.outerHTML
    }
    return x
  }).join(" ")
  showDecision({
    decision: state.decision,
    original: state.sentences[state.current_sentence],
    contradiction: state.contradiction,
    alternative: state.alternative,
    mitigation: state.mitigation,
    step: state.step,
    frame: i,
    explanation: state.explanation,
    triple: state.triple,
  })
  element.querySelectorAll(".mitigation-finished").forEach((e) => e.addEventListener('click', (e2) => showRevision(e2.target, i)))
  const speed = SPEEDS[PLAYER_SPEED]
  return ((state.generating === "sentences") ? speed.sentences : (state.status === "done" ? speed.done : speed.explanation))
}

function streamAnswer(stream, element, streamId, i=0){
  console.log(i)
  console.log(stream.length)
  document.querySelectorAll("input.slider").forEach(e => {e.max = stream.length-1; e.value = i})
  const delay = showState(stream, element, i)
  if (i+1 < stream.length && CURRENT_STREAM_ID === streamId && PLAYER_STATUS === PLAY){
    setTimeout(() => streamAnswer(stream, element, streamId, i+1), delay)
  }
  if (i+1 == stream.length && CURRENT_STREAM_ID == streamId){
    PLAYER_STATUS = PAUSE
  }
}

function selectLastRevision(){
  document.querySelectorAll(".side-by-side.selected").forEach(e => {
    Array.from(e.querySelectorAll("span.mitigation-finished.was-revised")).slice(-1)[0].click()
  })
}


// select example
window.addEventListener('load', function() {
  function switchToExample() {
    CURRENT_STREAM_ID += 1
    this.parentElement.childNodes.forEach(e => !e.classList || e == this || e.classList.remove('active'))
    this.classList.add('active')
    let selected = this.getAttribute('value')
    document.querySelectorAll('.side-by-side').forEach(e => {
      e.style.display = 'none';
      e.classList.remove('selected')
    });
    let id = selected.substr("--static-precomputed-".length)
    id = id.substr(0, id.length-"-jsonl".length)
    document.querySelectorAll('#' + selected).forEach(e => {
      e.style.display = 'flex';
      e.classList.add('selected');
      if(id != "live"){
        setTimeout(() => {
          PLAYER_STATUS = PAUSE;
          if (!textStream[selected]){
            fetch(`./static/precomputed/${id}.jsonl`).then(
              (res) => res.text()
            ).then(
              (resText) => {
            let prevMessage = ""
            const dmp = new diff_match_patch()
            const splitStream = resText.split("\n").filter(e => e)
            const currentStream = []
            for(let newMessage of splitStream){
              const patches = JSON.parse(newMessage)
              const restoredMessage = dmp.patch_apply(patches, prevMessage)[0]
              currentStream.push(JSON.parse(restoredMessage))
              prevMessage = restoredMessage
            }
            textStream[selected] = currentStream
            streamAnswer(currentStream, e, CURRENT_STREAM_ID, currentStream.length-1)
            selectLastRevision()
          })

          } else {
            streamAnswer(textStream[selected], e, CURRENT_STREAM_ID, textStream[selected].length-1)
            selectLastRevision()
          }
        })
      }
      else {
        showDecision(null)
      }
    });
    if (id == "chapais") {
      // remove hash
      if (window.location.hash != "") {
        history.pushState('', document.title, window.location.pathname);
      }
    } else {
      window.location.hash = id
    }
    let that = this;

    function timeTravelExample(event){
      PLAYER_STATUS = PAUSE
      x = event
      document.querySelectorAll('#' + selected).forEach(e => {
        showState(textStream[event.target.getAttribute("dataid")], e, Number(event.target.value))
      })

    }

    document.querySelectorAll("input.slider").forEach(e => e.addEventListener('input', timeTravelExample))
  }
    document.querySelectorAll("span.option.example").forEach(e => e.addEventListener('click', switchToExample))
  })

  window.addEventListener('load', function() {
    let id = window.location.hash
    if(!id || id === "#about" || id === "#learn-more" || id.startsWith("#hallucination-")){
      id = "#chapais"
    }
    id = id.slice(1)
    id = `--static-precomputed-${id}-jsonl`
    window.setTimeout(function() {
      // find first <anchor> in example element
      document.querySelectorAll(`span.option.example[value=${id}]`).forEach((e) => e.click())
    }, 100)
})

let screenshotMode = false;

function setScreenshotMode() {
  screenshotMode = true;
  document.body.classList.add("screenshot-mode")

  let clickDownY = null;
  let clickTop = null;
  let movingEl = null;
  
  document.querySelectorAll("anchor>label .multiline").forEach(e => {
    // on left click position relative top -1pt
    // on right click position relative top 1pt

    e.addEventListener('mousedown', function(event) {
      if (event.button == 0) {
        clickDownY = event.clientY;
        clickTop = parseInt(e.style.top || "0")
        movingEl = e
        event.stopPropagation()
      }
    })
  })

  document.body.addEventListener('mousemove', function(event) {
    if (clickDownY) {
      let delta = event.clientY - clickDownY;
      movingEl.style.top = (clickTop + delta) + "pt"
    }
  })

  document.body.addEventListener('mouseup', function(event) {
    if (clickDownY) {
      event.stopPropagation()
      let delta = event.clientY - clickDownY;
      movingEl.style.top = (clickTop + delta) + "pt"
      clickDownY = null;
      clickTop = null;
      movingEl = null;
    }
  })
}


// user input
window.addEventListener('load', function() {

  function handleUserPrompt(input){
    if(input.trim() in predefinedPrompts){
      const id = predefinedPrompts[input.trim()]
      document.querySelectorAll(`span.option.example[value=${id}]`).forEach((e) => e.click())
      return;
    }
    CURRENT_STREAM_ID += 1
    const ownStreamId = CURRENT_STREAM_ID
    document.querySelectorAll("#user-input").forEach(e => {
      e.innerText = input
      e.style["display"] = "block"
    })
    const element = document.querySelector("#--static-precomputed-live-jsonl")
    // fetch stream data from server and display
    const server = document.location.hostname == "localhost" ? "http://localhost:9113" : "https://api.chatprotect.ai"
    fetch(`${server}/chat?q=${encodeURIComponent(input)}`).then(
      (response) => {
        PLAYER_SPEED = 2
        let currentlyProcessing = ""
        let currentStream = []
        const streamReader = response.body.getReader()
        document.querySelectorAll("#user-output").forEach(e => {
          e.style["display"] = "block"
        })
        let prevMessage = ""
        const dmp = new diff_match_patch()

        streamReader.read().then(function process(r) {
          if (r.done) return;
          currentlyProcessing += new TextDecoder().decode(r.value)
          const splitStream = currentlyProcessing.split("\n")
          function recShow(stream){
            if (splitStream.length <= 1 || CURRENT_STREAM_ID != ownStreamId) {
              streamReader.read().then(process)
              return
            }
            const newMessage = splitStream.shift()
            // remove the message and  \n from the buffer
            currentlyProcessing = currentlyProcessing.slice(newMessage.length + 1)
            const patches = JSON.parse(newMessage)
            const restoredMessage = dmp.patch_apply(patches, prevMessage)[0]
            const pos = currentStream.push(JSON.parse(restoredMessage))
            prevMessage = restoredMessage
            const delay = showState(currentStream, element, pos-1)
            setTimeout(() => recShow(stream), delay)
          }
          recShow(splitStream)
        })
      
    })

  }
  function handleInput(inputField){
    if (!inputField.value) return;
    document.querySelectorAll("#user-output").forEach(e => {
      e.style["display"] = "none"
      e.innerText = ""
    })
    handleUserPrompt(inputField.value) 
    inputField.value = ""
  }
  
  document.querySelectorAll("#user-prompt").forEach(e => e.addEventListener('keyup', function(e){
    if (e.key === 'Enter' || e.keyCode === 13) {
      handleInput(e.target)
    }
  }))
  document.querySelectorAll("#user-prompt").forEach(e => e.addEventListener('click', function(e){
    // The user-send button is hidden behind the input and cannot be clicked so we detect whether the text input was clicked in the area of the button
    const sendButton = document.querySelector("#user-send")
    const sendButtonBox = sendButton.getBoundingClientRect()
    if (e.pageX >= sendButtonBox.x) handleInput(e.target)
  }))

})

// play controls
window.addEventListener('load', function() {

  document.querySelectorAll("#player-playpause").forEach(e => e.addEventListener('click', function(e){
    if(PLAYER_STATUS == PAUSE){
      PLAYER_STATUS = PLAY;
      CURRENT_STREAM_ID += 1
      const selectedElement = document.querySelector(".option.example.active")
      const selected = selectedElement.getAttribute("value")
      if(PLAYER_POSITION == textStream[selected].length-1) PLAYER_POSITION = 0;
      document.querySelectorAll('#' + selected).forEach(e => {
        streamAnswer(textStream[selected], e, CURRENT_STREAM_ID, PLAYER_POSITION)
      })
    }
    else {
      CURRENT_STREAM_ID += 1
      PLAYER_STATUS = PAUSE;
    }
  }))
  document.querySelectorAll("#player-stop").forEach(e => e.addEventListener('click', function(e){
    PLAYER_STATUS = PAUSE
    PLAYER_POSITION = 0
    setTimeout(() => {
      PLAYER_STATUS = PAUSE
      PLAYER_POSITION = 0
      CURRENT_STREAM_ID += 1
      const selectedElement = document.querySelector(".option.example.active")
      const selected = selectedElement.getAttribute("value")
      document.querySelectorAll('#' + selected).forEach(e => {
        streamAnswer(textStream[selected], e, CURRENT_STREAM_ID, PLAYER_POSITION)
      })
    }, 100)
  }))

  document.querySelectorAll("#player-forward").forEach(e => e.addEventListener('click', function(e){
    PLAYER_SPEED = (PLAYER_SPEED + 1) % SPEEDS.length
  }))
  document.querySelectorAll("#player-end").forEach(e => e.addEventListener('click', function(e){
    PLAYER_STATUS = PLAY
    const selectedElement = document.querySelector(".option.example.active")
    const selected = selectedElement.getAttribute("value")
    const stream = textStream[selected]
    PLAYER_POSITION = stream.length - 2
    CURRENT_STREAM_ID += 1
    setTimeout(() => {
      PLAYER_STATUS = PLAY
      PLAYER_POSITION = stream.length - 2
      document.querySelectorAll('#' + selected).forEach(e => {
        streamAnswer(stream, e, CURRENT_STREAM_ID, PLAYER_POSITION)
      })
    }, 100)
  }))
})

function findLastRevision(stream){
  for(let j = stream.length-1; j >= 0; j--){
    if(
      (stream[j].decision == "drop" || stream[j].decision == "redact")
    ) {
    console.log(stream[j])
     return j;
    }
  }
  return stream.length-1;
}