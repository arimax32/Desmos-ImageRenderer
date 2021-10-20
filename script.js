let elt = document.getElementById('calculator');
let calculator = Desmos.GraphingCalculator(elt);

function draw() {
	xhr = new XMLHttpRequest();
	xhr.open("GET", "http://127.0.0.1:5000/");
	xhr.send();

	xhr.onload = function() {
		contours = JSON.parse(xhr.response);
		let line = ""

		for(let i=0; i < contours.length; i++){

			let c = contours[i].length 
			for(let j=0;j < c - 1; j++){
				let x1 = contours[i][j][0][0]
				let y1 =-contours[i][j][0][1]
				let x2 = contours[i][j+1][0][0]
				let y2 =-contours[i][j+1][0][1]
				let m = (y2-y1)/(x2-x1)

				if (x1!==x2){
					if(y1<y2){
						line='y'+'-'+y1.toString()+'='+m.toString()+'('+'x'+'-'+x1.toString()+')'+'\\left\\{'+y1.toString()+'<'+'y'+'<'+y2.toString()+'\\right\\}'
					}
					if (y1>y2){
						line='y'+'-'+y1.toString()+'='+m.toString()+'('+'x'+'-'+x1.toString()+')'+'\\left\\{'+y2.toString()+'<'+'y'+'<'+y1.toString()+'\\right\\}'
					}
				}

				if(x1==x2 && y1<y2){
					line='x'+'='+x1.toString()+'\\left\\{'+y1.toString()+'<'+'y'+'<'+y2.toString()+'\\right\\}'
				}
				if(x1==x2 && y1>y2){
					line='x'+'='+x1.toString()+'\\left\\{'+y2.toString()+'<'+'y'+'<'+y1.toString()+'\\right\\}'
				}

		        calculator.setExpression({ id: `line${c*j + i + 1}`, latex: line, color: '#000000' , lineStyle: Desmos.Styles.SOLID }) //hex code of your prefered colour in color section
		    }  	//SOLID->solid lines, DASHED->dashed lines, DOTTED-> dotted lines ....
		}
	}
	return 'Its Working!'
}
draw()