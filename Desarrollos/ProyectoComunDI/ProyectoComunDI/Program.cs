using AccesoSqlServer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utilidades;

namespace ProyectoComunDI
{
    class Program
    {
        static void Main(string[] args)
        {


            #region Acceso MySQL

            #endregion
            #region AccesoCSV

			#endregion

			#region Configurador

			// aquí se instancian las clases necesarias para conectarse a cada base de datos.
			#endregion

            #region Aceso SQLServer
            //Lista de evidencias            
            List<Evidencia> evidencias = Consulta.ConsultaTodaLaTabla();

            //Se recorre la lista de objetos para extraer los datos
            foreach (Evidencia evidencia in evidencias)
            {
                
                /* Sale por consola todos los datos
                 * Console.WriteLine(evidencia.Longitud_petalo+"--"+evidencia.Longitud_sepalo+
                    "--"+evidencia.Ancho_petalo + "--" + evidencia.Ancho_sepalo+"--"+evidencia.Clase);*/
            }
            Console.ReadKey();

            #endregion

        }
    }

}
