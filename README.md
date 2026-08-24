LLM-prosjekt

Formålet med prosjektet er å kode og trene en fungerende tekst-prediktor i vanilla python + numpy, i tilsvarende nevralt nettverk-arkitektur som brukt i LLMs. Prosjektet er basert på teori og grunn-mal gitt i TMA4320 i 2024 (https://wiki.math.ntnu.no/_media/tma4320/2024v/prosjektbeskrivelse_01.03.pdf).

Modellen bruker embedding av tekst i en vektorrepresentasjon, som kjøres gjennom L lag som hvert kombinerer attention med feed-forward activation. Activation-funksjonen som tas i bruk er ReLu. Loss måles i crossentropy.

Modellarkitekturen er bygget ut gjennom en Layer-klasse, hvor funksjonsstegene (attention, softmax, embedding, ReLu etc) er child-klasser av denne. Disse har tilhørende metoder for forwards og backwards pass. Det benyttes en generisk LinearLayer klasse som håndterer tensorbehandling i de øvrige lagene.

Prosjektet benytter standard AD (automatic differentiation), og parametere lagres i dict-strukturer sammen med tilhørende gradienter. Ved trening av nettverket kjøres forward og backwards pass etterfølgende gjennom lagene, hvor løpende funksjonsverdier og gradienter tas vare på fortløpende for optimert kjøretid - som er industristandard.

Mål for prosjektet er i første rekke å trene modellen til å håndtere addisjon av små tall og sortering av korte strenger. Videre planlegging inkluderer å teste primtallsfaktorisering og standard tekstprediksjon.
