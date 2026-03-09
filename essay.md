# Engineering Computational Science

# Structural note

This document is the 'essay version' of my Inaugural lecture, typed in to help facilitate getting the length and content right, and verify the structure of the argument.

It will be supplemented by some code files for generating the exemplars, and eventually by a for-presentation version in Quarto.

# Part one: An exploration of scientific programming

## 1.1 A child's introduction to programming gravity

It's the mid-1980s. A BBC micro starts - Beep Boop!. I'm an early teenager, and I've just been reading some stuff about gravity. The explanation sort of makes sense, but I like programming
on my new-ish home computer, and I feel like there's bits of it I haven't understood.

I start trying to programme the computer to run a gravity simulation. 

Now, during this lecture, there will be some actual programmable demos I've built, but the code on the screen won't be real code - it'll be text designed to pull out the key aspects I'm trying to explain, that looks a bit like code. (We call it "pseudocode"). 

The demos are all generated using real code - and are available online for those who would like to look at them later - not in BBC basic but in Python so there's some utility for a modern audience.

So, kid-me has been told the rules of gravity work like this:

* Accelerate - change your velocity - toward anything heavy
  * How much? Well, the heavier it is, the more you change your speed
    * And it's 'proportional' - so twice as heavy means twice as much change
    * And it goes down with how far away from the 'sun' you are, 
        * proportional to how big the surface of the sphere is between you and the sun (square of radius) 
        * (does this mean the gravity is 'spreading out' so there's the same amount over the sphere?!?)
  * What does it mean 'toward' anything heavy?
    * Well, it's a vector - your velocity is a speed in a particular direction
    * And the change is in the direction of the sun or whatever.

How do we make that in code?

We need to start by examining 'change your velocity'.

We'll work in 2-d, with x and y coordinates. 

Suppose there's no gravity, and we have some speed, and the current speed is vx metres per second in the x direction, and vy metres per second in the y direction.

Then our code looks like

```
Let's have a time step dt, and make it small, so dt = 0.1
Each time step:
    x = x + vx * dt
    y = y + vy * dt
    Draw it
And that's the end of the time step, so go round and do it again.
```

If we show this animation, we get a dot, moving in a straight line! [Demo]

In order to type this in, we have to have started developing understandings of scalar and vector quantities, and child-me has made, based on the bit of calculus that he's starting to know, some kind of guess at
how small time steps should enable us to implement velocity as change in position - how many metres do you go in a given amount of time.

Note the "loop", a bit  we do the "each time step" bit over and over again, and we indent 

Now we can put in a sun at `(x=0, y=0)`, and add gravity:

```
dt = 0.1
loop:
    distance_to_sun = square_root (x^2 + y^2) -- Pythagoras!
    amount_to_accelerate = gravity_strength_constant * sun_mass / distance_to_sun^2
    x_share_of_acceleration = x / distance_to_sun
    y_share_of_acceleration = y / distance_to_sun
    vx = vx + amount_to_accelerate * x_share_of_acceleration
    vy = vy + amount_to_acceerlate * y_share_of_acceleration
    x = x + vx * dt
    y = y + vy * dt
end of loop
```

We type this in, and watch the video, and it looks like it's working! [Demo]

The lesson here that I really want to get across here is that for me, and for many of us, the process of building the code, is a really useful way of making sure **there's none of the science you don't understand**.
I definitely didn't understand this properly as a kid, but the point of doing this WASN'T to build a gravity simulation. It was to **use the process of programming as a tool to explore whether you've really understood the thing you're trying to put in the computer**. We call this "modelling" - the computer contains a 'model' of the earth going round the sun.

But then we draw the trajectory over time - and instead of being a circle, it doesn't quite meet up! Something is wrong! Oh no, did we misunderstand gravity!?!?! [Figure]

Now, probably quite a few of you in this audience will know that the problem here is the *approach to implementing the calculus* - the small timestep approach we took is called "Forward Euler Integration", and it works quite badly. After asking some annoying questions, this really blew me away as a kid. We followed the 'obvious' approach to implementing the meaning of velocities and accelerations, and it sort of worked, but sort of didn't. A quick trip to Kendal library and an implementation of Runge-Kutte helped sort that out.

So this is another aspect of our lesson about what we learn by programming in science: effectively implementing computer programmes to model the world introduces us to interesting new challenges, in ways that we didn't expect.

### 1.1.2 - Computational science beyond physics

Another thing tenage me really liked a few years later - I think I saw this in one of my Dad's books but then did an implentation on Archimedes - is that pretty similar code to this can be used - with a different rule for acceleration - to simulate behaviour of flocking sheep. (Remember I'm from Cumbria.) Or indeed flocking birds - with really beautiful videos. 

You can code in rules for formation flying, avoiding predators etc. [Demo should show the Boids]

The rules for the Boids


This also really made an impression on me - the kind of understanding-with-rules, where you aren't properly convinced you've understood something until you can get a computer to do it, works in biology and sociology, and history, as well as physics. 



My Dad bought a copy of a book called 'frontiers of complexity' just as I was heading off to uni - I would later have the author of that book as a postdoc supervisor, using computer simulations to model blood flow in the brain. [Wave to Peter.]



# 1.2 Speed - readability tradeoffs

Some of you will have noticed that the gravity programme is inefficient:

```
    distance_to_sun = square_root (x^2 + y^2) -- Pythagoras!
    amount_to_accelerate = gravity_strength_constant * sun_mass / distance_to_sun^2
```

could be:

```
    amount_to_accelerate = gravity_strength_constant / (x^2 + y^2)
```

...we've saved ourself doing a square root and then a square. Do we want to make that change? The new programme will be faster, but it will be harder to understand. Does that matter?
Well, what if we want to change the programme, or give it to someone else to make changes? My grandad was also into this kind of thing, and it was nice to be able to show him the code.

Let's take this a step further - we're way beyond my teenage example now - but it's another useful bit of the exploration. Suppose we need to simulate lots of gravitating things - maybe a galaxy or something -
and we have to use lots of computers together to solve it. We call these 'compute clusters' or 'supercomputers'. They need to pass messages between each other, like this:

```
On each computer:
   Loop:
      Do-the-gravity-thing
        And take into account the particles on the other computers, but 'smudged up' rather than one by one.
      For any particles getting to the edge of "my bit":
        Pass them to the computer responsible for the part of the world it's going to.
      Receive from the other computers:
        Any particles getting to me I need to take care of
        And the information about the 'smudged up' mass
```

(In doing this, there's some nice maths about how to do the 'smudging up' about why it's safe, but we're focusing on the computing here.)

[Demo should show a particle being passed over between two simulation domains, and some pretty results for a big system]

We got a big speed-up from using a supercomputer, but there's LOTS AND LOTS of new complexity. Do our networks work for passing all that information around? How do we do the book keeping of who takes charge of which particle? How do we divide up space between the computers? Does everyone need to talk to everyone? Or can I get away with only talking to my neighbours? Otherwise the programme will spend *all it's effort in conversations and no time actually doing gravity maths*.

It's a LOT harder to write the programme now. And a lot easier to make mistakes.

This is our second deep lesson: **because in science we use code to understand the world, but the calculations are expensive, the trade off between lucidity and speed really matters**. (We usually say 'complexity and performance'.)

# 1.3 Making mistakes, science, and what is it all for?

So this is where this lecture starts to take a philosophical turn. It'll be quite wordy for a while, but for those who like that better, we'll get back to some more examples of results from calculations and code later.

Why does lucidity in code matter so much

Boids were done for graphics. Engineering helps science and vice versa.

# 1.5 Conclusion of part one - Engineering Computational Science

# Part two: Epistemologies, viewpoints, and careers

# Part three: Some futures

Pedagogical programming in a world of AI.