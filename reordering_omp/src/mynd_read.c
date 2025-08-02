#ifndef READ_H
#define READ_H

#include "mynd_functionset.h"

/*************************************************************************
* This function checks if a file exists
**************************************************************************/
reordering_int_t mynd_Is_file_exists(char *fname)
{
    struct stat status;

    if (stat(fname, &status) == -1)
        return 0;

    return S_ISREG(status.st_mode);
}

/*************************************************************************
* This function opens a file
**************************************************************************/
FILE * mynd_check_fopen(char *fname, char *mode, const char *message)
{
    FILE *fp = fopen(fname, mode);
    if (fp != NULL)
        return fp;

    char *error_message = (char *)mynd_check_malloc(sizeof(char) * 1024, "check_fopen: error_message");
	sprintf(error_message, "Failed on check_fopen()\n\tfile: %s, mode: %s, [ %s ].", fname, mode, message);
    mynd_error_exit(error_message);

    return NULL;
}

/*************************************************************************/
/*! This function is the GKlib implementation of glibc's getline()
    function.
    \returns -1 if the EOF has been reached, otherwise it returns the 
             number of bytes read.
*/
/*************************************************************************/
ssize_t  mynd_check_getline(char **lineptr, reordering_int_t *n, FILE *stream)
{
    reordering_int_t i;
    reordering_int_t ch;

    /* Check whether the file stream reaches the end of the file, and if it does, return -1 */
    if (feof(stream))
        return -1;  

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0) {
        *n = 1024;
        *lineptr = (char *)mynd_check_malloc(sizeof(char) * (*n), "check_getline: lineptr");
    }

    /* get into the main loop */
    i = 0;
    /* The getc function is used to read characters from the file stream until the end of the file is reached */
    while ((ch = getc(stream)) != EOF) {
        (*lineptr)[i++] = (char)ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i+1 == *n) { 
            *n = 2*(*n);
            *lineptr = (char *)mynd_check_realloc(*lineptr, (*n)*sizeof(char), (*n)*sizeof(char)/2, "check_getline: lineptr");
        }
        
        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
}

/*************************************************************************
* This function closes a file
**************************************************************************/
void  mynd_check_fclose(FILE *fp)
{
	fclose(fp);
}

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
/*
params->filename = graphfile

*/
void mynd_ReadGraph(char *filename, reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t **txadj, reordering_int_t **tvwgt, reordering_int_t **tadjncy, reordering_int_t **tadjwgt)
{
    reordering_int_t i, k, l, fmt, nfields, readew, readvw, readvs, edge, ewgt;
    reordering_int_t *xadj, *adjncy, *vwgt, *adjwgt;
	reordering_int_t *vsize;
    char *line = NULL, fmtstr[256], *curstr, *newstr;
    reordering_int_t lnlen = 0;
    FILE *fpin;
    graph_t *graph;

    if (!mynd_Is_file_exists(filename)) 
    {
        // char *error_message = (char *)malloc(sizeof(char) * 128);   //!!!
        char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "File %s does not exist!", filename);
        mynd_error_exit(error_message);
        // errexit("File %s does not exist!\n", params->filename);
    }
	
    graph = mynd_CreateGraph();

    fpin = mynd_check_fopen(filename, "r", "ReadGRaph: Graph");
	
    /* Skip comment lines until you get to the first valid line */
    do {
        if ( mynd_check_getline(&line, &lnlen, fpin) == -1) 
        {
            char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
			sprintf(error_message, "Premature end of input file: file: %s.", filename);
            mynd_error_exit(error_message);
        }
    } while (line[0] == '%');

    fmt = 0;
    nfields = sscanf(line, "%"PRIDX" %"PRIDX" %"PRIDX"", &(graph->nvtxs), &(graph->nedges), &fmt);
	
	if (nfields < 2) 
	{
		char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "The input file does not specify the number of vertices and edges.");
		mynd_error_exit(error_message);
	}
	
	if (graph->nvtxs <= 0 || graph->nedges <= 0) 
	{
		char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "The supplied nvtxs: %"PRIDX" and nedges: %"PRIDX" must be positive.", graph->nvtxs, graph->nedges);
		mynd_error_exit(error_message);
	}
	
	if (fmt > 111) 
	{
		char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "Cannot read this type of file format [fmt=%"PRIDX"]!", fmt);
		mynd_error_exit(error_message);
	}
	
	sprintf(fmtstr, "%03"PRIDX"", fmt%1000);
	readvs = (fmtstr[0] == '1');
	readvw = (fmtstr[1] == '1');
	readew = (fmtstr[2] == '1');
	// printf("readvs=%"PRIDX" readvw=%"PRIDX" readew=%"PRIDX"\n",readvs, readvw, readew);
	
	// if (ncon > 0 && !readvw) 
	// {
	// 	char *error_message = (char *)mynd_check_malloc(sizeof(char) * 1024, "ReadGraph: error_message");
	// 	sprintf(error_message, 
	// 	"------------------------------------------------------------------------------\n"
	// 	"***  I detected an error in your input file  ***\n\n"
	// 	"You specified ncon=%d, but the fmt parameter does not specify vertex weights\n" 
	// 	"Make sure that the fmt parameter is set to either 10 or 11.\n"
	// 	"------------------------------------------------------------------------------\n", ncon);
	// 	mynd_error_exit(error_message);
	// }

	graph->nedges *=2;
	// ncon = graph->ncon = (ncon == 0 ? 1 : ncon);

	// xadj   = graph->xadj   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (graph->nvtxs + 1), "ReadGraph: xadj");
	// adjncy = graph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "ReadGraph: adjncy");
	// vwgt   = graph->vwgt   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "ReadGraph: vwgt");
	// adjwgt = graph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges,"ReadGraph: adjwgt");
	xadj   = *txadj   = (reordering_int_t *)malloc(sizeof(reordering_int_t) * (graph->nvtxs + 1));
	adjncy = *tadjncy = (reordering_int_t *)malloc(sizeof(reordering_int_t) * graph->nedges);
	vwgt   = *tvwgt   = (reordering_int_t *)malloc(sizeof(reordering_int_t) * graph->nvtxs);
	adjwgt = *tadjwgt = (reordering_int_t *)malloc(sizeof(reordering_int_t) * graph->nedges);

	// vsize  = graph->vsize  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs,"ReadGraph: vsize");

	mynd_set_value_int(graph->nvtxs + 1, 0, xadj);
	mynd_set_value_int(graph->nvtxs, 1, vwgt);
	mynd_set_value_int(graph->nedges, 1, adjwgt);
	// mynd_set_value_int(graph->nvtxs, 1, vsize);

	/*----------------------------------------------------------------------
	* Read the sparse graph file
	*---------------------------------------------------------------------*/
	for (xadj[0] = 0, k = 0, i = 0; i < graph->nvtxs; i++) 
	{
		do {
			if ( mynd_check_getline(&line, &lnlen, fpin) == -1) 
			{
				char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "Premature end of input file while reading vertex %"PRIDX".", i + 1);
				mynd_error_exit(error_message);
			}
		} while (line[0] == '%');

		curstr = line;
		newstr = NULL;

    	/* Read vertex sizes */
		if (readvs) 
		{
			vsize[i] = strtoll(curstr, &newstr, 10);
			if (newstr == curstr)
			{
				char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "The line for vertex %"PRIDX" does not have vsize information.", i + 1);
				mynd_error_exit(error_message);
			}
			if (vsize[i] < 0)
			{
				char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "The size for vertex %"PRIDX" must be >= 0.", i + 1);
				mynd_error_exit(error_message);
			}

			curstr = newstr;
		}


		/* Read vertex weights */
		if (readvw) 
		{
			for (l=0; l<1; l++) 
			{
				vwgt[i+l] = strtoll(curstr, &newstr, 10);
				if (newstr == curstr)
				{
					char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The line for vertex %"PRIDX" does not have enough weights for the %"PRIDX" constraint.", i + 1, l);
					mynd_error_exit(error_message);
				}
				if (vwgt[i + l] < 0)
				{
					char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The weight vertex %"PRIDX" and constraint %"PRIDX" must be >= 0.", i + 1, l);
					mynd_error_exit(error_message);
				}

				curstr = newstr;
			}
		}

		while (1) 
		{
			edge = strtoll(curstr, &newstr, 10);
			if (newstr == curstr)
				break; /* End of line */
			curstr = newstr;

			// jiusuo
			// edge++;

			if (edge < 1 || edge > graph->nvtxs)
			{
				char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "Edge %"PRIDX" for vertex %"PRIDX" is out of bounds.", edge, i + 1);
				mynd_error_exit(error_message);
			}

			ewgt = 1;
			if (readew) 
			{
				ewgt = strtoll(curstr, &newstr, 10);
				if (newstr == curstr)
				{
					char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "Premature end of line for vertex %"PRIDX".", i + 1);
					mynd_error_exit(error_message);
				}
				if (ewgt <= 0)
				{
					char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The weight (%"PRIDX") for edge (%"PRIDX", %"PRIDX") must be positive.", ewgt, i + 1, edge);
					mynd_error_exit(error_message);
				}

				curstr = newstr;
			}

			if (k == graph->nedges)
			{
				char *error_message = (char *)mynd_check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "There are more edges in the file than the %"PRIDX" specified.", graph->nedges / 2);
				mynd_error_exit(error_message);
			}

			adjncy[k] = edge-1;
			adjwgt[k] = ewgt;
			k++;
		}

    	xadj[i + 1] = k;
	}

	mynd_check_fclose(fpin);

	if (k != graph->nedges) 
	{
		printf("------------------------------------------------------------------------------\n");
		printf("***  I detected an error in your input file  ***\n\n");
		printf("In the first line of the file, you specified that the graph contained\n"
			"%"PRIDX" edges. However, I only found %"PRIDX" edges in the file.\n", 
			graph->nedges / 2, k / 2);
		if (2 * k == graph->nedges) 
		{
			printf("\n *> I detected that you specified twice the number of edges that you have in\n");
			printf("    the file. Remember that the number of edges specified in the first line\n");
			printf("    counts each edge between vertices v and u only once.\n\n");
		}
		printf("Please specify the correct number of edges in the first line of the file.\n");
		printf("------------------------------------------------------------------------------\n");
		exit(0);
	}

	// printf("lnlen=%zu sizeof(char) * lnlen=%zu\n",lnlen, sizeof(char) * lnlen);

	mynd_check_free(line, sizeof(char) * lnlen, "ReadGraph: line");
	nvtxs[0]  = graph->nvtxs;
	nedges[0] = graph->nedges;
	
	// return graph;
}

#endif