function add_js(js_file){
    var script_el = document.createElement("script");
    script_el.setAttribute("type", "text/javascript");
    script_el.setAttribute("src", js_file);
    document.getElementsByTagName("head")[0].appendChild(script_el);
}
function add_css(css_file){
    var link_el = document.createElement("link");
    link_el.setAttribute("rel", "stylesheet");
    link_el.setAttribute("type", "text/css");
    link_el.setAttribute("href", css_file);
    document.getElementsByTagName("head")[0].appendChild(link_el);
}