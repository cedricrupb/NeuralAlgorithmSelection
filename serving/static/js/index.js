
let options = {
  name: 'breadthfirst',

  fit: true, // whether to fit the viewport to the graph
  directed: true, // whether the tree is directed downwards (or edges can point in any direction if false)
  padding: 30, // padding on fit
  circle: false, // put depths in concentric circles if true, put depths top down if false
  grid: true, // whether to create an even grid into which the DAG is placed (circle:false only)
  spacingFactor: 1.75, // positive spacing factor, larger => more space between nodes (N.B. n/a if causes overlap)
  boundingBox: undefined, // constrain layout bounds; { x1, y1, x2, y2 } or { x1, y1, w, h }
  avoidOverlap: true, // prevents node overlap, may overflow boundingBox if not enough space
  nodeDimensionsIncludeLabels: false, // Excludes the label when calculating node bounding boxes for the layout algorithm
  roots: undefined, // the roots of the trees
  maximal: false, // whether to shift nodes down their natural BFS depths in order to avoid upwards edges (DAGS only)
  animate: false, // whether to transition the node positions
  animationDuration: 500, // duration of animation in ms if enabled
  animationEasing: undefined, // easing of animation if enabled,
  animateFilter: function ( node, i ){ return true; }, // a function that determines whether the node should be animated.  All nodes animated by default on animate enabled.  Non-animated nodes are positioned immediately when the layout starts
  ready: undefined, // callback on layoutready
  stop: undefined, // callback on layoutstop
  transform: function (node, position ){ return position; } // transform a given node position. Useful for changing flow direction in discrete layouts
};

var cyto;
var select_options;
var markers = [];
var attention_marker = [];
var attention_buffer;


function predict(){
  reset();

  dim = $('#recommend .dimmer');
  dim.addClass("active");
  dim.empty();
  dim.html("<div class=\"ui loader\"></div>");
  $('#prediction').text("Prediction is not finished.");

  button = $("#send");

  if(button.hasClass("loading")){
    failMsg("You have to wait till request is finished.");
    return;
  }

  var dataset = $('#dataset .text').text();
  var model = $('#model .menu .active').attr("data-value");
  var id = select_options[dataset][model][0];
  var category = $('#category .text').text();

  var text = editor.getValue();

  $.ajax({
    url: '/api/task/',
    type: 'PUT',
    data: {data: text, model_id: id, category: category},
    success: function(data){
      startProcess(data);
    }
  }).fail(function(){
    failMsg("Something went wrong will connection.");
  });

}

function handleUpdate(id, data){
  reset();
  if('exception' in data){
    failMsg(data['exception']);
  }else{
    if(data['graph']){
      loadGraph(id);
    }

    if(data['position']){
      loadPosition(id);
    }

    if(data['pred']){
      loadPrediction(id);
    }

    if(data['attention']){
      loadAttention(id);
    }
  }
  if(data['finish']){
    $('#send').removeClass('loading');
  }
}

function statusUpdates(id){
  var finished = false;
  $.ajax({
    url: 'api/task/'+id+"/",
    type: 'GET',
    success: function(data){
      handleUpdate(id, data);
      finished = data['finish'];
    },
    complete: function(){
      if(!finished){
        setTimeout(function(){statusUpdates(id);}, 500);
      }
    }
  })

}

function startProcess(data){
  id = data['request_id'];
  $('#send').addClass('loading');
  statusUpdates(id);
}

function failMsg(msg){
  fc = $("#fail");
  $("#fail .header").text(msg);
  fc.removeClass('hidden');
}

function reset(){
  fc = $("#fail");
  if(!fc.hasClass("hidden")){
    fc.addClass("hidden");
  }

  for(am of attention_marker){
    am.clear();
  }
  attention_marker = [];

}

function presentGraph(id, data){
  container = $('#graph_container');
  container.parent().parent().children(".dimmer").removeClass(
    "active"
  );
  cyto.elements().remove();

  var ele = [];

  for(var key in data['nodes']){
    if(key.includes("_"))continue;
    name = data['nodes'][key];
    ele.push({
      group: 'nodes', data: {id: key, label: name}
    });
  }

  i = 0;
  colors = {};
  for(edge of data["edges"]){
    u = edge[0];
    v = edge[2];
    if(u.includes("_") || v.includes("_"))continue;
    type = edge[1];
    color = '#ccc';

    if(type.startsWith("cd")){
      color = '#4cbb17';
    }else if(type.startsWith("dd")){
      color = '#ff5349';
    }

    colors["e_"+i] = color;
    ele.push({
      group: 'edges', data: {
        id: "e_"+i, source: u, target: v, type: type
      }
    });
    i++;
  }

  cyto.add(ele);

  for(e in colors){
    color = colors[e];
    cyto.$id(e).style("line-color", color);
    cyto.$id(e).style('curve-style', 'bezier');
  }

  layout = cyto.layout(options);
  layout.run();
}

function attachPostion(id, data){
  for(key in data){
    node = cyto.$id(key);
    if(node != undefined){
      node.data('pos', data[key]);
    }
  }
}

function loadPosition(id){
  container = $('#graph_container');
  if(container.attr('graph') != id)
    return;

  if(container.attr('pos') == id)
    return;

  container.attr('pos', id);

  $.ajax({
    url:"/api/position/"+id+"/",
    type: 'GET',
    success: function(data){attachPostion(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load position...");
      container.removeAttr('pos');
    }
  )
}

function loadGraph(id){
  container = $('#graph_container');
  if(container.attr('graph') == id)
    return;

  container.attr('graph', id);

  $.ajax({
    url:"/api/graph/"+id+"/",
    type: 'GET',
    success: function(data){presentGraph(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load graph...");
      container.removeAttr('graph');
    }
  )
}

function presentPrediction(id, data){
  s = "";
  for(var i = 0; i < data.length && i < 8; i++){
    s = s + " > "+data[i];
  }
  s = s.substring(3);
  $('#prediction').text(s);
}

function loadPrediction(id){
  container = $('#prediction');
  if(container.attr('graph') == id)
    return;
  container.attr('graph', id);

  $.ajax({
    url:"/api/prediction/"+id+"/",
    type: 'GET',
    success: function(data){presentPrediction(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load prediction...");
      container.removeAttr('graph');
    }
  )
}


function attentionToGraph(data){
  atts = {}
  for(att of data){
    atts[att[3]] = att[2];
  }

  max = 0.0;
  min = 1.0;
  for(a in atts){
    v = atts[a];
    if(v > max){
      max = v;
    }
    if(v < min){
      min = v;
    }
  }

  for(a in atts){
      attention = (atts[a] - min)/(max - min);
      node = cyto.$id(a);
      if(attention >= 0.5){
        node.style('background-color', "rgb(0, 100, 0)");
        node.style('background-opacity', attention);
      }else{
        node.style('background-color', "#ccc");
        node.style('background-opacity', 1.0);
      }
  }

}


function presentAttention(pos){
  data = attention_buffer[pos];
  lineAttention = {};

  for(m of attention_marker){
    m.clear();
  }
  attention_marker = [];
  for(m of markers){
    m.clear();
  }
  markers = [];

  attentionToGraph(data);

  for(att of data){
    pos0 = att[0];
    pos1 = att[1];

    if(pos1 - pos0 > 0){
      continue;
    }

    a = att[2];

    for(p = pos0; p <= pos1; p++){
      pos = p;
      if(pos <= 1){
        pos = 1;
      }
      pos = p - 1;
      if(!(pos in lineAttention)){
        lineAttention[pos] = 0.0;
      }
      lineAttention[pos] += a;
    }
  }

  max = 0.0;
  min = 1.0;
  for(key in lineAttention){
    v = lineAttention[key];
    if(v > max){
      max = v;
    }
    if(v < min){
      min = v;
    }
  }

  for(key in lineAttention){
    attention = (lineAttention[key] - min)/(max - min);

    if(attention >= 0.1){
      line = editor.getLine(key);
      if(line != undefined){
        pos0 = CodeMirror.Pos(key, 0);
        pos1 = CodeMirror.Pos(key, line.length);
        attention_marker.push(editor.markText(
          pos0, pos1, {css: "background-color: rgba(0, 100, 0, "+attention+");color:white;"}
        ));
      }
    }

  }

}

function bufferAttention(id, data){
  layer_menu = $('#layer');
  layer_menu.removeClass('disabled');
  layer_menu.children('.menu').empty();

  attention_buffer = data;

  i = 0;
  small = 100;
  for(att of data){
    if(att != undefined){
      layer_menu.children('.menu').append(
        "<div class=\"item\" data-value=\""+i+"\">Layer "+i+"</div>"
      );
      if(i < small){
        small = i;
      }
    }
    i++;
  }

  if(small < 100){
    layer_menu.children('.text').removeClass('default');
    layer_menu.children('.text').text('Layer '+small);
    presentAttention(small);

    layer_menu.dropdown({
      'set selected': small,
      onChange: function(value, text, $selectItem){
        presentAttention(value);
      }
    });
  }

}

function loadAttention(id){
  container = $('#prediction');
  if(container.attr('attention') == id)
    return;
  container.attr('attention', id);

  $.ajax({
    url:"/api/attention/"+id+"/",
    type: 'GET',
    success: function(data){bufferAttention(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load attention...");
      container.removeAttr('attention');
    }
  )
}

function enable_predict(){
  $('#send').removeClass('disabled');
}

function build_suboption(base_src){
  $('#model .menu').empty();

  for(key in select_options[base_src]){
    score = select_options[base_src][key][1];
    $("#model .menu").append("<div class=\"item\" data-value=\""+key+"\">"+key+": "+score.toFixed(3)+"</div>");
  }

  $('#model').removeClass("disabled");
  $('#model').dropdown(
    {
      onChange: function(value, text, $selectItem){
        enable_predict();
      }
    }
  );
}


function buildOptions(data){
  select_options = data;

  for(key in data){
    $("#dataset .menu").append("<div class=\"item\" data-value=\""+key+"\">"+key+"</div>");
  }

  $('#dataset').dropdown(
    {
      onChange: function(value, text, $selectItem){
        build_suboption(value);
      }
    }
  );
}


function addOptions(){
  $.get("/api/models/", success=buildOptions).fail(
    function(){
      failMsg("Cannot load options for prediction.")
    }
  );
}

function highlightPosition(id){
  node = cyto.$id(id);
  pos = node.data('pos');

  for(m of attention_marker){
    m.clear();
  }
  attention_marker = [];
  for(m of markers){
    m.clear();
  }
  markers = [];

  if(pos != undefined){

    pos0 = pos[0];
    pos1 = pos[1];

    for(var p = pos0; p <= pos1; p++){
      pos = p;
      if(pos <= 1){
        pos = 1;
      }
      pos = pos - 1;
      p0 = CodeMirror.Pos(pos, 0);
      p1 = CodeMirror.Pos(pos, 100);
      markers.push(editor.markText(
        p0, p1, {css: "background-color: blue; color: white;"}
      ));
    }

  }
}


function initSite(){
  container = $('#graph_container');
  $('#fail .close').on('click', function(){
    $(this).closest('#fail')
           .transition('fade');
  });
  cyto = cytoscape({
    container: container,
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)'
        }
      },
      {
        selector: 'edge',
        style: {
          'line-color': '#ccc',
          'target-arrow-color': '#ccc',
          'target-arrow-shape': 'triangle'
        }
      }
    ]
  });
  cyto.on('click', 'node', function(evt){
      highlightPosition(this.id());
  });
  editor.on('change', function(cm, change){
    for(am of attention_marker){
      am.clear();
    }
    attention_marker = [];
  })
  $('#category').dropdown();
  addOptions();
}
