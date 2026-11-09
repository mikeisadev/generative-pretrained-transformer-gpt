/**
 * Vista "Pesi numerici": la registrazione LEGGERA. La vista compare subito nel pannello Viste e nei link
 * diretti, ma il suo codice (three.js per i numeri sulle matrici, la tabella e l'istogramma condivisi, il
 * run "GPT del corso" con gli snapshot, l'albero dei moduli: una ventina di moduli) si carica soltanto
 * alla prima apertura (numbersView.js). Chi non la apre non lo scarica e l'avvio del tab non rallenta.
 *
 *   installNumbersView(tab);                         // in index.js, con le altre viste
 *   await tab.numbersView.open({ tensor, step });   // goto, ricerca: carica il modulo e apre la vista
 *
 * Appena il modulo è caricato tab.numbersView diventa l'API completa (vedi numbersView.js).
 */

export const NUMBERS_VIEW_ID = "numeri";
/** Link "i pesi che nascono durante il training": la vista sul training, dallo step 0. */
export const BIRTH_VIEW_ID = "numeri.training";

export function installNumbersView(tab) {
  let impl = null; // la vista vera, quando il modulo è caricato
  let loading = null;
  let wanted = false; // la vista è (ancora) quella scelta: uno studente veloce può lasciarla prima del caricamento
  const load = () =>
    (loading ??= import("./numbersView.js")
      .then((m) => {
        impl = m.createNumbersView(tab);
        tab.numbersView = impl.api;
        return impl;
      })
      .catch((err) => {
        loading = null; // un errore di rete non resta in cache: si riprova alla prossima apertura
        throw err;
      }));

  tab.views.register({
    id: NUMBERS_VIEW_ID,
    label: "Pesi numerici",
    description: "I numeri veri dentro le matrici: avvicinati e i colori diventano numeri, anche negli step del training.",
    order: 15,
    enter: async () => {
      wanted = true;
      if (!impl) tab.setCaption("Pesi numerici: preparo i numeri del modello…");
      let view;
      try {
        view = await load();
      } catch (err) {
        console.error(err);
        if (wanted) tab.setCaption(`Impossibile caricare la vista Pesi numerici: ${err.message}`);
        return;
      }
      if (wanted && tab.views.current === NUMBERS_VIEW_ID) await view.enter();
    },
    exit: () => {
      wanted = false;
      impl?.exit();
    },
    // "Centra" e "Vista d'insieme": il tensore scelto (prima del caricamento la pila intera)
    home: () => (impl ? impl.home() : tab.focus("overview")),
  });

  // API leggera fino al caricamento: basta per i link diretti e la ricerca
  tab.numbersView = {
    loaded: false,
    open: async (options) => (await load()).api.open(options),
  };
}
